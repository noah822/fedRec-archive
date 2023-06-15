import torch
from io import BytesIO
from typing import Dict, Any
'''
    Mediate through torch store/load api
'''

def serialize(data: Dict[str, Any]) -> bytes:
    bytes_io = BytesIO()
    torch.save(data, bytes_io)
    return bytes_io.getvalue()

def deserialize(stream: bytes) -> Dict[str, Any]:
    bytes_io = BytesIO(stream)
    unpacked = torch.load(bytes_io)
    return unpacked
