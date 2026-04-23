import panel as pn
from pathlib import Path

RESULTS_DIR = Path("/data/chet-cussp/results")
IMAGES = {
    "Raypath properties projected onto fault plane": RESULTS_DIR / "raypath_fault_plane.png",
    "ML workflow 3D trajectory": RESULTS_DIR / "ml_trajectory_3d.png",
}


def image_column(caption, path):
    if path.exists():
        return pn.Column(
            pn.pane.Markdown(f"### {caption}"),
            pn.pane.PNG(str(path), sizing_mode="scale_width"),
            sizing_mode="stretch_width",
        )
    return pn.Column(
        pn.pane.Markdown(f"### {caption}"),
        pn.pane.Alert(
            f"Image not yet available: **{path.name}**  \n"
            f"Place the PNG at `{path}`",
            alert_type="warning",
        ),
        sizing_mode="stretch_width",
    )


def build_layout():
    panels = [image_column(caption, path) for caption, path in IMAGES.items()]
    return pn.Row(*panels, sizing_mode="stretch_width")


pn.extension()

pn.template.VanillaTemplate(
    title="CUSSP Results",
    logo="/home/chopp/CUSSP.png",
    main=pn.Column(
        pn.pane.Markdown(
            "Fault characterization and ML workflow outputs from the CUSSP EGS Collab experiment.",
            sizing_mode="stretch_width",
        ),
        build_layout(),
        sizing_mode="stretch_width",
    ),
).servable()
