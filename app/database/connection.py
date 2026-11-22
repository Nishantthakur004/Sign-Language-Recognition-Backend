from __future__ import annotations

import logging
from typing import Any

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo import ASCENDING, DESCENDING
from pymongo.errors import PyMongoError

from app.config import Settings

logger = logging.getLogger(__name__)


def create_client(settings: Settings) -> AsyncIOMotorClient:
    return AsyncIOMotorClient(
        settings.mongodb_uri,
        serverSelectionTimeoutMS=1500,
        connectTimeoutMS=1500,
        socketTimeoutMS=10000,
        maxPoolSize=50,
        uuidRepresentation="standard",
    )


async def configure_indexes(db: AsyncIOMotorDatabase) -> None:
    await db.recognition_history.create_index([("user_id", ASCENDING), ("created_at", DESCENDING)])
    await db.recognition_history.create_index([("user_id", ASCENDING), ("sign", ASCENDING)])
    await db.recognition_history.create_index([("user_id", ASCENDING), ("session_id", ASCENDING)])
    await db.recognition_history.create_index(
        [("user_id", ASCENDING), ("prediction_id", ASCENDING)],
        unique=True,
        partialFilterExpression={"prediction_id": {"$type": "string"}},
    )
    await db.favourites.create_index([("user_id", ASCENDING), ("sign_id", ASCENDING)], unique=True)


async def ping_database(db: AsyncIOMotorDatabase) -> bool:
    try:
        await db.command("ping")
        return True
    except (PyMongoError, AttributeError, NotImplementedError):
        return False


async def startup_database(app: Any, settings: Settings) -> None:
    client = create_client(settings)
    db = client[settings.mongodb_db]
    app.state.mongo_client = client
    app.state.db = db
    app.state.db_indexes_ready = False
    app.state.db_available = await ping_database(db)
    if app.state.db_available:
        try:
            await configure_indexes(db)
        except PyMongoError:
            logger.exception("MongoDB connected but index creation failed")
            app.state.db_available = False
        else:
            app.state.db_indexes_ready = True
    else:
        logger.warning("MongoDB is unavailable; database-backed endpoints will return 503")


async def shutdown_database(app: Any) -> None:
    client = getattr(app.state, "mongo_client", None)
    if client is not None:
        client.close()
