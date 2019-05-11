# import os
# REDIS_URL = "redis://{host}:{port}/1".format(
#     host=os.getenv('REDIS_HOST', 'localhost'),
#     port=os.getenv('REDIS_PORT', '6379')
# )
# CELERY_BROKER_URL=REDIS_URL
# CELERY_RESULT_BACKEND=REDIS_URL
CELERY_BROKER_URL='redis://redis-service:6379/0'
CELERY_RESULT_BACKEND='redis://redis-service:6379/0'
