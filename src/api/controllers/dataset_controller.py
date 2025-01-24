from fastapi import APIRouter, Response

router = APIRouter()

@router.get('')
def get_datasets():
    """
    Responsible for looking for the available models.
    :return: the list of available models
    """

    try:
        # Return the list of available datasets
        pass
    except ApplicationError as e:
        return APIResponse(status_code=e.status_code, details=e.get_message())
    except Exception as e:
        return APIResponse(status_code=500, details=e.__str__())