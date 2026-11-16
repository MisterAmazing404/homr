"""Odyssey entry point: run homr on one page and return MusicXML plus positions.

The MusicXML carries `imgpos` and `odyssey` comments on notes and `odyssey-bar` comments on
measures (see music_xml_generator.py). The staff geometry is returned separately because
it has no home in MusicXML.
"""

import xml.etree.ElementTree as ET
from dataclasses import dataclass

from homr import main as homr_main
from homr.music_xml_generator import TimeSignature, XmlGeneratorArguments, generate_xml
from homr.odyssey_align import OdysseyContext, StaffGeometry
from homr.onnx_providers import coreml_available, cuda_available, rocm_available
from homr.simple_logging import eprint
from homr.staff_parsing import parse_staffs
from homr.system_repair import repair_systems
from homr.transformer.configs import Config


@dataclass
class PageResult:
    musicxml: str
    geometry: list[StaffGeometry]
    last_times: list[TimeSignature | None]


def _processing_config() -> homr_main.ProcessingConfig:
    transformer_gpu = cuda_available() or rocm_available()
    segnet_gpu = transformer_gpu or coreml_available()
    return homr_main.ProcessingConfig(
        enable_debug=False,
        enable_cache=False,
        write_staff_positions=False,
        read_staff_positions=False,
        selected_staff=-1,
        transformer_use_gpu=transformer_gpu,
        segnet_use_gpu=segnet_gpu,
        coreml_encoder=False,
        title_detection=False,
    )


def init_models() -> None:
    """Download the model weights if they are missing."""
    config = _processing_config()
    homr_main.download_weights(config.segnet_use_gpu, config.transformer_use_gpu, False)


def run_page(
    image_path: str, inherited_times: list[TimeSignature | None] | None = None
) -> PageResult:
    config = _processing_config()
    single = homr_main.detect_single_staffs(image_path, config)
    multi_staffs = homr_main.group_staffs(single.predictions, single.staffs)
    eprint("Found", len(multi_staffs), "systems:", [len(m.staffs) for m in multi_staffs])
    multi_staffs = repair_systems(single.debug, multi_staffs)

    transformer_config = Config()
    transformer_config.use_gpu_inference = config.transformer_use_gpu
    context = OdysseyContext(single.bar_lines)
    voices = parse_staffs(
        single.debug,
        multi_staffs,
        single.predictions.preprocessed,
        selected_staff=-1,
        config=transformer_config,
        page_to_input_image=single.to_input_image,
        odyssey=context,
    )
    xml = generate_xml(XmlGeneratorArguments(False, None, None), voices, "", inherited_times)
    single.debug.clean_debug_files_from_previous_runs()
    from homr.music_xml_generator import last_time_signatures  # noqa: PLC0415

    return PageResult(
        musicxml=ET.tostring(xml, encoding="unicode"),
        geometry=context.geometry,
        last_times=last_time_signatures(xml),
    )
