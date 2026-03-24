import os
import sys
from grpc_tools import protoc
import importlib.resources

def build_protobufs():
    # Find the main project folder no matter where this script is run from
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    # Define where the blueprint is
    protos_dir = os.path.join(project_root, 'protos')
    proto_file = os.path.join(protos_dir, 'inference.proto')

    # Single output location — the shared inference-protos package
    target_folder = os.path.join(project_root, 'inference-protos', 'inference_protos')
    proto_include = str(importlib.resources.files('grpc_tools') / '_proto')

    print(f"Translating data blueprints from {proto_file}...")

    os.makedirs(target_folder, exist_ok=True)

    result = protoc.main([
        'grpc_tools.protoc',
        f'-I{protos_dir}',
        f'-I{proto_include}',       # add this line
        f'--python_out={target_folder}',
        proto_file
    ])

    if result == 0:
        print(f" -> Successfully compiled into: {target_folder}")
    else:
        print(f" -> Error compiling protos", file=sys.stderr)
        sys.exit(1)

    print("\nAll done! Run 'pip install inference-protos/' in each node to update.")

if __name__ == '__main__':
    build_protobufs()