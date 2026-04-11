def is_retryable_grpc_error(exception):

    import grpc

    if isinstance(exception, grpc.aio.AioRpcError):
        return exception.code() in {
            grpc.StatusCode.UNAVAILABLE,
            grpc.StatusCode.DEADLINE_EXCEEDED,
            grpc.StatusCode.RESOURCE_EXHAUSTED,
        }

    if isinstance(exception, (ConnectionRefusedError, OSError)):
        return True

    return False
