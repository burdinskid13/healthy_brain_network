import click

@click.command()
@click.option("--cachedir")


def run(
    cachedir='/om2/user/maedbh/.cache/pydra-ml/cache-wf/'
    ):
    import shutil

    # delete cache directory
    shutil.rmtree(cachedir)


if __name__ == "__main__":
    run()
