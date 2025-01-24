from fastapi import APIRouter, Response

router = APIRouter()

"""
Gets the state of the current API

Returns
-------
Response Code
    200
"""
@router.get('')
async def check_health() -> Response:
    return Response(status_code=200)