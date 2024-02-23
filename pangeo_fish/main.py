import rich_click as click

click.rich_click.USE_MARKDOWN = True
click.rich_click.SHOW_ARGUMENTS = True


@click.group()
def main():
    """Run the pangeo-fish model."""
    pass


@main.command(
    "prepare",
    short_help="transform the data into something suitable for the model to run",
)
def prepare():
    pass


@main.command("estimate", short_help="estimate the model parameter")
def estimate():
    pass


@main.command("decode", short_help="produce the model output")
def decode():
    pass
