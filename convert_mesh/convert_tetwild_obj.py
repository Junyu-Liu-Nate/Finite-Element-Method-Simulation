def convert_mesh_to_obj(input_filename, output_filename):
    with open(input_filename, 'r') as file:
        lines = file.readlines()
    
    vertices_section = False
    tetrahedra_section = False
    vertices = []
    tetrahedra = []
    
    for line in lines:
        line = line.strip()
        if "Vertices" in line:
            vertices_section = True
            tetrahedra_section = False
            continue
        
        if "Tetrahedra" in line:
            vertices_section = False
            tetrahedra_section = True
            continue
        
        if "End" in line:
            break
        
        if vertices_section and line and not line.isdigit():
            vertex_data = line.split()[:3]
            vertices.append(vertex_data)
                
        if tetrahedra_section and line and not line.isdigit():
            tetrahedron_data = line.split()[:4]
            if tetrahedron_data:
                tetrahedra.append(tetrahedron_data)
                
    # Writing to the new OBJ format
    with open(output_filename, 'w') as outfile:
        for v in vertices:
            outfile.write(f'v {" ".join(v)}\n')
        
        # OBJ files typically use faces (f) instead of tetrahedra (t)
        # Here we convert tetrahedra to faces, assuming that each tetrahedron
        # can be represented by its four triangular faces
        for t in tetrahedra:
            # Note: OBJ indices are 1-based, so we adjust from 0-based without subtracting 1
            outfile.write(f'f {int(t[0])} {int(t[1])} {int(t[2])}\n')
            outfile.write(f'f {int(t[0])} {int(t[1])} {int(t[3])}\n')
            outfile.write(f'f {int(t[0])} {int(t[2])} {int(t[3])}\n')
            outfile.write(f'f {int(t[1])} {int(t[2])} {int(t[3])}\n')

# Example usage
input_filename = '/Users/liujunyu/Downloads/ftetwild_output_msh/32770.stl_0121.mesh'
output_filename = '/Users/liujunyu/Downloads/32770.stl_0121.obj'
convert_mesh_to_obj(input_filename, output_filename)
