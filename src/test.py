from merlin.systems.triton.utils import send_triton_request
from merlin.core.dispatch import make_df
import numpy as np
from nvtabular import ColumnSchema, Schema

# create a request to be sent to TIS
request = make_df({"user_id": [7]})
request["user_id"] = request["user_id"].astype(np.int32)
print(request)

request_schema = Schema(
    [
        ColumnSchema("user_id", dtype=np.int32),
    ]
)
output_list = ['ordered_ids', 'ordered_scores']
response = send_triton_request(request_schema, request, output_list)
print(response)