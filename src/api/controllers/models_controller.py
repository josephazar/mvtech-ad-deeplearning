from fastapi import APIRouter, Response
from gi.overrides.Gio import Application

router = APIRouter()

@router.get('/models')
def get_models():

    """
    Responsible for looking for the available models.
    :return: the list of available models
    """

    try:
        # Return the list of available models
        pass
    except ApplicationError as e:
        return APIResponse(status_code=e.status_code, details=e.get_message())
    except Exception as e:
        return APIResponse(status_code=500, details=e.__str__())