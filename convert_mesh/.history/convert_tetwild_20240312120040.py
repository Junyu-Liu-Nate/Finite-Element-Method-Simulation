def convert_mesh(input_filename, output_filename):
    with open(input_filename, 'r') as file:
        lines = file.readlines()
    
    vertices_section = False
    tetrahedra_section = False
    vertices = []
    tetrahedra = []
    
    for line in lines:
        if "Vertices" in line:
            vertices_section = True
            tetrahedra_section = False
            continue
        
        if "Tetrahedra" in line:
            vertices_section = False
            tetrahedra_section = True
            continue
        
        if vertices_section and not line.strip().isdigit():
            # Split the line, take the first 3 components, and ignore the last one
            vertex_data = line.strip().split()[:3]
            if vertex_data:
                vertices.append(vertex_data)
                
        if tetrahedra_section and not line.strip().isdigit():
            # Split the line, take the first 4 components, and ignore the last one
            tetrahedron_data = line.strip().split()[:4]
            if tetrahedron_data:
                tetrahedra.append(tetrahedron_data)
                
    # Writing to the new .mesh format
    with open(output_filename, 'w') as outfile:
        for v in vertices:
            outfile.write(f'v {" ".join(v)}\n')
        for t in tetrahedra:
            # Adjust indices to be 1-based to 0-based
            t_adjusted = [str(int(i)-1) for i in t]
            outfile.write(f't {" ".join(t_adjusted)}\n')

# Example usage
input_filename = '/Users/liujunyu/Downloads/ftetwild_output_msh/32770.stl_0121.mesh'
output_filename = '/Users/liujunyu/Downloads/ftetwild_output_msh/32770.stl_0121.mesh'
convert_mesh(input_filename, output_filename)


# Example usage
msh_filename = '/Users/liujunyu/Downloads/ftetwild_output_msh/32770.stl_0121.msh'
mesh_filename = '/Users/liujunyu/Downloads/ftetwild_output_msh/32770.stl_0121.mesh'
convert_msh_to_mesh(msh_filename, mesh_filename)
