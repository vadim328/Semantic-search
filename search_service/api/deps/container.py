from fastapi import Request
from search_service.container.di import Container


def get_container(request: Request) -> Container:
    return request.app.state.container
