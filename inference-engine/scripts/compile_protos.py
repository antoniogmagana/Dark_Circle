import os
import sys
from grpc_tools import protoc

def build_protobufs():
    # Find the main project folder no matter where this script is run from
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    # Define where the blueprint is
    protos_dir = os.path.join(project_root, 'protos')
    proto_file = os.path.join(protos_dir, 'inference.proto')

    # List of the folders that need a copy of the Python data classes
    target_folders = [
        os.path.join(project_root, 'src', 'ingestor'),
        os.path.join(project_root, 'src', 'infer_detect'),
        os.path.join(project_root, 'src', 'infer_classify'),
        os.path.join(project_root, 'src', 'egress')
    ]

    print(f"Translating data blueprints from {proto_file}...")

    # Loop through each folder and build a copy directly into it
    for folder in target_folders:
        # Make sure the folder actually exists first
        os.makedirs(folder, exist_ok=True)
        
        # Run the compiler tool for this specific folder
        result = protoc.main([
            'grpc_tools.protoc',
            f'-I{protos_dir}',
            f'--python_out={folder}',
            proto_file
        ])

        if result == 0:
            print(f" -> Successfully placed files in: {folder}")
        else:
            print(f" -> Error building files for {folder}", file=sys.stderr)
            sys.exit(1)
            
    print("\nAll done! Every node now has its own copy of the data classes.")

if __name__ == '__main__':
    build_protobufs()