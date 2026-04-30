import os
from upstash_redis import AsyncRedis

# Redis client instance
redis_client = None


async def init_redis():
    """Initialize Upstash Redis client"""
    global redis_client
    redis_url = os.getenv("UPSTASH_REDIS_URL")
    redis_token = os.getenv("UPSTASH_REDIS_TOKEN")

    if not redis_url or not redis_token:
        raise ValueError("UPSTASH_REDIS_URL and UPSTASH_REDIS_TOKEN environment variables must be set")

    redis_client = AsyncRedis(url=redis_url, token=redis_token)


async def close_redis():
    """Close Redis connection"""
    global redis_client
    # Upstash Redis client doesn't require explicit closing
    redis_client = None


def get_redis():
    """Dependency to get Redis client"""
    return redis_client


async def validate_user_id(user_id: str) -> bool:
    """
    Validate if user_id exists in Redis registered users.

    Uses EXISTS on a keyed pattern for O(1) lookup instead of scanning all keys.
    Returns True if valid, False otherwise.
    """
    redis = get_redis()
    # Use EXISTS for O(1) lookup with key pattern valid_user:{user_id}
    exists = await redis.exists(f"valid_user:{user_id}")
    return bool(exists)


async def migrate_user_ids_to_keys():
    """
    One-time migration function to create valid_user:{user_id} keys.

    Scans existing Redis hashes and creates validation keys for all hashed_id values.
    This should be run once to migrate from the old validation approach.
    """
    redis = get_redis()

    print("Starting migration: scanning existing Redis keys...")
    keys = await redis.keys("*")

    user_ids_found = []
    for key in keys:
        # Skip validation keys that already exist
        if key.startswith("valid_user:"):
            continue

        key_type = await redis.type(key)
        if key_type == "hash":
            stored_user_id = await redis.hget(key, "hashed_id")
            if stored_user_id:
                user_ids_found.append(stored_user_id)

    if user_ids_found:
        # Create validation key for each user ID
        for user_id in user_ids_found:
            await redis.set(f"valid_user:{user_id}", "1")
        print(f"Migration complete: Created {len(user_ids_found)} validation keys")
    else:
        print("Migration complete: No user IDs found")

    return len(user_ids_found)


async def add_user_id_key(user_id: str):
    """
    Add a validation key for a user_id.

    This should be called whenever a new user is registered externally.
    External systems that create user hashes must also call this function
    or directly execute: SET valid_user:{user_id} 1
    """
    redis = get_redis()
    await redis.set(f"valid_user:{user_id}", "1")


async def remove_user_id_key(user_id: str):
    """
    Remove a validation key for a user_id.

    This should be called if a user is deactivated or removed.
    """
    redis = get_redis()
    await redis.delete(f"valid_user:{user_id}")