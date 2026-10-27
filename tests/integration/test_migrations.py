import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from kip.db.models import Base
from sqlalchemy import inspect
from sqlalchemy.orm import Session, sessionmaker


@pytest.mark.postgres
def test_migration_matches_models_and_can_downgrade(sessions: sessionmaker[Session]) -> None:
    engine = sessions.kw["bind"]
    if engine.dialect.name != "postgresql":
        pytest.skip("the production migration targets PostgreSQL")
    config = Config("alembic.ini")
    with engine.begin() as connection:
        assert not compare_metadata(MigrationContext.configure(connection), Base.metadata)
        config.attributes["connection"] = connection
        command.downgrade(config, "base")
        assert set(inspect(connection).get_table_names()) == {"alembic_version"}
        command.upgrade(config, "head")
        assert not compare_metadata(MigrationContext.configure(connection), Base.metadata)
