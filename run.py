import click
from distro import name
from backend import data_prep
from backend import generateEmbeddings 
from backend import chat

@click.group()
def cli():
    pass

@click.command('fetch_data')
def fetch_data():
    click.echo('Initialized the database')
    data_prep.main()
    click.echo('Data preparation complete')

@click.command('create_embeddings')
def create_embeddings():
    click.echo('Created embeddings for the dataset')
    generateEmbeddings.main()
    click.echo('Embedding creation complete')

@click.command('query')
def query():
    click.echo('Starting query with the model')
    chat.main()
    click.echo('Query session ended')


cli.add_command(fetch_data)
cli.add_command(create_embeddings)
cli.add_command(query)

if __name__ == '__main__':
    cli()