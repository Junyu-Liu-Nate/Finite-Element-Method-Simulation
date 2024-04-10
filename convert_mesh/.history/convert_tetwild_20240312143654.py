def convert_mesh(input_filename, output_filename):
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
        
        # Check for the 'End' marker and stop processing if found
        if "End" in line:
            break
        
        if vertices_section and line and not line.isdigit():
            vertex_data = line.split()[:3]
            vertices.append(vertex_data)
                
        if tetrahedra_section and line and not line.isdigit():
            tetrahedron_data = line.split()[:4]
            if tetrahedron_data:
                tetrahedra.append(tetrahedron_data)
                
    # Writing to the new .mesh format
    with open(output_filename, 'w') as outfile:
        for v in vertices:
            outfile.write(f'v {" ".join(v)}\n')
        for t in tetrahedra:
            # Adjust indices to be 1-based to 0-based
            # t_adjusted = [str(int(i)-1) for i in t]
            t_adjusted = [str(int(i)) for i in t]
            outfile.write(f't {" ".join(t_adjusted)}\n')

# Example usage
input_filename = '/Users/liujunyu/Downloads/alien.mesh'
output_filename = '/Users/liujunyu/Downloads/alien_converted.mesh'
convert_mesh(input_filename, output_filename)
