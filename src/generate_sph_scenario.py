from pathlib import Path
import re
import shutil


PROJECT_ROOT = Path(r"C:\JalRaksha")

BASE_XML = (
    PROJECT_ROOT
    / "simulations"
    / "dualsphysics"
    / "dam_break_validation"
    / "CaseDambreakVal2D_Def.xml"
)

SCENARIO_DIR = (
    PROJECT_ROOT
    / "simulations"
    / "dualsphysics"
    / "scenarios"
)


def generate_scenario(
    scenario_name: str,
    reservoir_length_m: float,
    reservoir_height_m: float,
) -> Path:

    if not BASE_XML.exists():
        raise FileNotFoundError(f"Base XML not found: {BASE_XML}")

    output_dir = SCENARIO_DIR / scenario_name
    output_dir.mkdir(parents=True, exist_ok=True)

    output_xml = output_dir / f"{scenario_name}_Def.xml"

    xml = BASE_XML.read_text(encoding="utf-8")

    # Change the initial fluid column size.
    xml = re.sub(
        r'(<size x=")[^"]+(" y="2" z=")[^"]+(" />)',
        rf'\g<1>{reservoir_length_m}\g<2>{reservoir_height_m}\g<3>',
        xml,
        count=1,
    )

    output_xml.write_text(xml, encoding="utf-8")

    # Keep the file-box definition available for later FlowTool processing.
    boxes = (
        PROJECT_ROOT
        / "simulations"
        / "dualsphysics"
        / "dam_break_validation"
        / "CaseDambreak_FileBoxes.txt"
    )

    if boxes.exists():
        shutil.copy2(boxes, output_dir / "CaseDambreak_FileBoxes.txt")

    print("Scenario created successfully.")
    print(f"Scenario: {scenario_name}")
    print(f"Reservoir length: {reservoir_length_m:.2f} m")
    print(f"Reservoir height: {reservoir_height_m:.2f} m")
    print(f"Definition: {output_xml}")

    return output_xml


if __name__ == "__main__":
    generate_scenario(
        scenario_name="scenario_02",
        reservoir_length_m=1.5,
        reservoir_height_m=2.0,
    )