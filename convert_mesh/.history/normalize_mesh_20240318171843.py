import numpy as np

def shrink_mesh_to_unit_sphere(vertices):
    # Convert vertices list to a NumPy array for easier manipulation
    vertices_array = np.array(vertices, dtype=float)
    
    # Step 1: Translate the mesh to center it at (0,0,0)
    centroid = np.mean(vertices_array, axis=0)
    centered_vertices = vertices_array - centroid
    
    # Step 2: Scale the mesh to fit inside a unit sphere
    max_distance = np.max(np.linalg.norm(centered_vertices, axis=1))
    scaled_vertices = centered_vertices * 2 / max_distance
    
    return scaled_vertices.tolist()

def read_and_shrink_mesh(input_filename):
    with open(input_filename, 'r') as file:
        lines = file.readlines()
    
    vertices = []
    for line in lines:
        if line.startswith('v '):
            _, x, y, z = line.split()
            vertices.append([float(x), float(y), float(z)])
    
    # Process the vertices to fit them within a unit sphere
    shrunk_vertices = shrink_mesh_to_unit_sphere(vertices)
    
    # Replace the original vertices in the file content with the shrunk ones
    for i, line in enumerate(lines):
        if line.startswith('v '):
            v_str = 'v ' + ' '.join(map(str, shrunk_vertices.pop(0))) + '\n'
            lines[i] = v_str
    
    return lines

# Example usage
input_filename = '/Users/liujunyu/Downloads/1772591.stl_0121_converted.mesh'
output_filename = '/Users/liujunyu/Downloads/1772591.stl_0121_converted_normalized.mesh'
shrunk_mesh_lines = read_and_shrink_mesh(input_filename)

with open(output_filename, 'w') as file:
    file.writelines(shrunk_mesh_lines)
