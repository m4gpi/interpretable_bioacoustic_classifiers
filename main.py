import click
import sys
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

@click.group()
def cli():
    pass

@cli.command()
@click.argument("args", nargs=-1)
def train(args):
    from src.cli.train import main
    sys.argv = ["train"] + list(args)
    main()

@cli.command()
@click.argument("args", nargs=-1)
def eval(args):
    from src.cli.eval import main
    sys.argv = ["eval"] + list(args)
    main()

@cli.command()
@click.argument("args", nargs=-1)
def app(args):
    from src.app.main import main
    sys.argv = ["app"] + list(args)
    main()

if __name__ == "__main__":
    cli()
