from .runner import MigrationRunner


def run_migrations(engine):
    MigrationRunner(engine).run_up()
