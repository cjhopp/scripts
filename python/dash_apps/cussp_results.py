import panel as pn

# Images are served as static files by NGINX at /result-images/<filename>
# Panel never reads these files — it just renders <img> tags.
IMAGES = {
    "ML Tomography": "/result-images/ML_tomo.png",
    "ML Trajectory": "/result-images/ML_trajectory.png",
}


def image_column(caption, url):
    return pn.Column(
        pn.pane.Markdown(f"### {caption}"),
        pn.pane.HTML(
            f'<img src="{url}" style="width:100%; height:auto;" '
            f'onerror="this.replaceWith(document.createTextNode(\'Image not yet available: {url}\'))" />'
        ),
        sizing_mode="stretch_width",
    )


def build_layout():
    panels = [image_column(caption, url) for caption, url in IMAGES.items()]
    return pn.Row(*panels, sizing_mode="stretch_width")


pn.extension()

pn.template.VanillaTemplate(
    title="CUSSP Results",
    logo="/home/chopp/CUSSP.png",
    main=pn.Column(
        pn.pane.Markdown(
            "Fault characterization and ML workflow outputs from the CUSSP experiment.",
            sizing_mode="stretch_width",
        ),
        build_layout(),
        sizing_mode="stretch_width",
    ),
).servable()
