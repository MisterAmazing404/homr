import json
import sys
import traceback

import cv2
import numpy as np

from homr import constants
from homr.bounding_boxes import BoundingBox, create_lines
from homr.debug import Debug
from homr.model import MultiStaff, Staff, StaffPoint
from homr.point_mapping import PointMapping, identity
from homr.simple_logging import eprint
from homr.staff_detection import detect_staff
from homr.type_definitions import NDArray

# The staff positions are saved in one of two formats, the file says which:
#
# - precise (JSON, what save_staff_positions writes): the staffs of every system with their
#   full grids and grand staff flags, in the pixels of the image the user supplied. A
#   later run loads them and doesn't detect or group staffs again.
# - boxes (text, one line per staff: class, centre x, centre y, width, height, all relative
#   to the image): the original coarse format. The staffs inside the boxes are
#   detected again, see load_staff_boxes. Reading it is kept, writing it is save_staff_boxes.
FORMAT = "homr-staves"
VERSION = 1


def save_staff_positions(
    multi_staffs: list[MultiStaff],
    file_path: str,
    to_input_image: PointMapping = identity,
    input_shape: tuple[int, ...] = (0, 0),
) -> None:
    """
    Saves the staffs exactly as they are: per system its staffs with every grid point
    (x, angle and the y of each line), whether it is a grand staff, and whether the staff
    relation model was sure of the system. to_input_image maps the staffs' coordinates to
    the image the user supplied, whose numpy shape is input_shape (height, width, ...).
    """
    systems = []
    for multi_staff in multi_staffs:
        staffs = []
        for staff in multi_staff.staffs:
            grid = []
            for point in staff.grid:
                mapped = [to_input_image((point.x, y)) for y in point.y]
                x = float(np.mean([m[0] for m in mapped]))
                grid.append([x, float(point.angle), *[float(m[1]) for m in mapped]])
            staffs.append({"grandstaff": bool(staff.is_grandstaff), "grid": grid})
        systems.append({"certain": bool(multi_staff.certain), "staffs": staffs})
    data = {
        "format": FORMAT,
        "version": VERSION,
        "image": {"width": int(input_shape[1]), "height": int(input_shape[0])},
        "systems": systems,
    }
    with open(file_path, "w") as file:
        json.dump(data, file, separators=(",", ":"))


def is_precise_staff_positions(file_path: str) -> bool:
    with open(file_path) as file:
        return file.read(1) == "{"


def load_precise_staff_positions(
    file_path: str,
    from_input_image: PointMapping = identity,
    input_shape: tuple[int, ...] | None = None,
) -> list[MultiStaff]:
    """
    The staffs as save_staff_positions saved them. from_input_image maps the image the
    user supplied to the one the staffs are wanted in, see point_mapping.invert.
    input_shape is the shape of the image the user supplied now, it must be the one the
    staffs were saved for.
    """
    with open(file_path) as file:
        data = json.load(file)
    if data.get("format") != FORMAT or data.get("version") != VERSION:
        raise ValueError("Not a staff positions file of this version: " + file_path)
    saved = data["image"]
    if input_shape is not None and (
        saved["width"] != input_shape[1] or saved["height"] != input_shape[0]
    ):
        raise ValueError(
            f"The staffs were saved for an image of {saved['width']}x{saved['height']}, "
            f"this one is {input_shape[1]}x{input_shape[0]}"
        )
    result = []
    for system in data["systems"]:
        staffs = []
        for saved_staff in system["staffs"]:
            grid = []
            for x, angle, *ys in saved_staff["grid"]:
                points = [from_input_image((x, y)) for y in ys]
                grid.append(
                    StaffPoint(
                        float(np.mean([p[0] for p in points])), [p[1] for p in points], angle
                    )
                )
            staff = Staff(grid)
            staff.is_grandstaff = saved_staff["grandstaff"]
            staffs.append(staff)
        result.append(MultiStaff(staffs, [], certain=system["certain"]))
    return result


def load_staff_positions(
    debug: Debug,
    image: NDArray,
    file_path: str,
    selected_staff: int = -1,
    from_input_image: PointMapping = identity,
    input_shape: tuple[int, ...] | None = None,
) -> list[MultiStaff]:
    """
    Loads staff positions in either format. image is the one the staffs are wanted in (only
    the boxes format needs it, to detect the staffs inside the boxes).
    """
    if is_precise_staff_positions(file_path):
        return load_precise_staff_positions(file_path, from_input_image, input_shape)
    return load_staff_boxes(debug, image, file_path, selected_staff)


def save_staff_boxes(
    multi_staffs: list[MultiStaff], shape: tuple[int, ...], file_path: str
) -> None:
    staff_coordinates = []
    for multi_staff in multi_staffs:
        for staff in multi_staff.staffs:
            x1, y1, x2, y2 = staff.min_x, staff.min_y, staff.max_x, staff.max_y
            width = x2 - x1
            height = y2 - y1
            centerx = x1 + width / 2
            centery = y1 + height / 2
            img_height, img_width = shape
            staff_class = "1" if staff.is_grandstaff else "0"
            coordinate = (
                staff_class
                + " "
                + str(centerx / img_width)
                + " "
                + str(centery / img_height)
                + " "
                + str(width / img_width)
                + " "
                + str(height / img_height)
            )
            staff_coordinates.append(coordinate + "\n")
    with open(file_path, "w") as text_file:
        text_file.writelines(staff_coordinates)


def load_staff_boxes(
    debug: Debug, image: NDArray, file_path: str, selected_staff: int = -1
) -> list[MultiStaff]:
    staffs = []
    img_height, img_width, _ = image.shape

    with open(file_path) as text_file:
        lines = text_file.readlines()

    for line_index, line in enumerate(lines):
        try:
            parts = line.strip().split()
            if len(parts) % constants.number_of_lines_on_a_staff != 0:
                continue  # Ignore invalid lines

            staff_class, norm_centerx, norm_centery, norm_width, norm_height = map(float, parts)
            is_grandstaff = staff_class != 0

            width = norm_width * img_width
            height = norm_height * img_height
            centerx = norm_centerx * img_width
            centery = norm_centery * img_height

            x1 = int(centerx - width / 2)
            y1 = int(centery - height / 2)
            x2 = int(x1 + width)
            y2 = int(y1 + height)

            bounding_box = BoundingBox([x1, y1, x2, y2], np.array([]))
            crop_area = bounding_box.increase_size_in_each_dimension(100, image.shape)
            staff_img = crop_area.blank_everything_outside_of_box(image)
            if selected_staff >= 0 and line_index != selected_staff:
                staff = dummy_staff_from_rect(bounding_box, image.shape)
                if staff is not None:
                    staff.is_grandstaff = is_grandstaff
                    staffs.append(MultiStaff([staff], []))
                continue

            staff = detect_staff_simple(debug, staff_img, crop_area)
            if staff is None:
                staff = dummy_staff_from_rect(bounding_box, image.shape)
            min_staff_width = 10
            if staff is not None and staff.max_x - staff.min_x > min_staff_width:
                staff.is_grandstaff = is_grandstaff
                staffs.append(MultiStaff([staff], []))
        except Exception as e:
            eprint("Skipping staff due to error:", e)
            traceback.print_exc(file=sys.stderr)

    return staffs


def detect_staff_simple(debug: Debug, img: NDArray, crop_area: BoundingBox) -> Staff | None:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5
    )

    # Create horizontal structuring element and apply morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 1))
    horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_ERODE, kernel, iterations=2)
    horizontal_lines = cv2.morphologyEx(horizontal_lines, cv2.MORPH_DILATE, kernel, iterations=2)
    horizontal_lines = cv2.morphologyEx(horizontal_lines, cv2.MORPH_CLOSE, kernel, iterations=2)
    debug.write_threshold_image("horizontal_lines", 255 * horizontal_lines)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 10))
    vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_ERODE, kernel, iterations=2)
    vertical_lines = cv2.morphologyEx(vertical_lines, cv2.MORPH_DILATE, kernel, iterations=2)
    vertical_lines = cv2.morphologyEx(vertical_lines, cv2.MORPH_CLOSE, kernel, iterations=2)
    debug.write_threshold_image("vertical_lines", 255 * vertical_lines)

    # Detect lines using Hough Transform
    staff_lines = create_lines(horizontal_lines, threshold=30, min_line_length=30)
    bar_lines = create_lines(vertical_lines, threshold=20, min_line_length=20, max_line_gap=15)
    staff_lines = [line.ensure_min_dimension(3, 3) for line in staff_lines]
    bar_lines = [line.ensure_min_dimension(3, 3) for line in bar_lines]
    bar_lines = [bar.make_box_taller_keep_center(5) for bar in bar_lines]

    crop_box = crop_area.box
    min_x = max(crop_box[0] + 100, 0)
    max_x = crop_box[3] - 100

    staffs = detect_staff(debug, horizontal_lines, staff_lines, [], bar_lines)
    if len(staffs) > 0:
        return staffs[0].extend_to_x_range(min_x, max_x)
    return None


def dummy_staff_from_rect(box: BoundingBox, shape: tuple[int, ...]) -> Staff | None:
    x1, y1, x2, y2 = box.box
    x1 = max(0, x1 - 50)
    x2 = min(shape[0], x2 + 50)
    points = []
    height = y2 - y1
    for x in range(int(x1), int(x2), 10):
        yValues = [(i * height / 4) + y1 for i in range(5)]
        points.append(StaffPoint(x, yValues, 0))
    if len(points) == 0:
        return None
    return Staff(points)


if __name__ == "__main__":
    import sys

    img = cv2.imread(sys.argv[1])
    if img is None:
        raise ValueError("Failed to read " + sys.argv[1])
    staff = detect_staff_simple(
        Debug(img, "input.jpg", True), img, BoundingBox([100, 100, 500, 500], np.array([]))
    )

    if staff is None:
        print("No staff lines detected.")  # noqa: T201
    else:
        print(staff)  # noqa: T201
