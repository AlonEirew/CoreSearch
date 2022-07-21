import uvicorn

from fastapi.logger import logger as fastapi_logger
from rest_api.utils import get_app, get_pipelines
import logging

gunicorn_error_logger = logging.getLogger("gunicorn.error")
gunicorn_logger = logging.getLogger("gunicorn")
uvicorn_access_logger = logging.getLogger("uvicorn.access")
uvicorn_access_logger.handlers = gunicorn_error_logger.handlers

fastapi_logger.handlers = gunicorn_error_logger.handlers
logging.handlers = gunicorn_error_logger.handlers

if __name__ != "__main__":
    fastapi_logger.setLevel(gunicorn_logger.level)
else:
    fastapi_logger.setLevel(logging.DEBUG)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

app = get_app()
pipelines = get_pipelines()  # Unused here, called to init the pipelines early


logger.info("Open http://127.0.0.1:8081/docs to see Swagger API Documentation.")
logger.info(
    """
    Or just try it out directly: curl --request POST --url 'http://127.0.0.1:8081/query' 
    -H "Content-Type: application/json"  --data '{"query": "Who is the father of Arya Stark?"}'
    """
)


if __name__ == "__main__":
    # logging.basicConfig(format="%(asctime)s %(message)s", datefmt="%m/%d/%Y %I:%M:%S %p")
    # gunicorn_logger = logging.getLogger('gunicorn.error')
    # logger.handlers = gunicorn_logger.handlers
    # logger.setLevel(gunicorn_logger.level)
    uvicorn.run(app, host="0.0.0.0", port=8081, debug=True)
