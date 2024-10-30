# Trabalho 2 - Computação Gráfica
# Alunos: Hiago Vinicius Americo - 11218469, Vítor Beneti Martins - 11877635

# A cena representa o sentimento de solidão, com uma casa isolada no meio do deserto.
# O vazio da cena traz uma sensação de solidão e isolamento, enquanto o cachorro solitário passa a sensação de abandono.


# Controles: WASD - Movimentos da câmera
# P - ativa/desativa o modo malha
# 1-7 - Seleciona o objeto
# R - ativa o modo de rotação
# T - ativa o modo de translação 
# Z/X - aumenta/diminui o tamanho do objeto selecionado
# Modo rotalçao:
# - seta para esquerda: diminui o angulo de rotação
# - seta para direita: aumenta o angulo de rotação
# Modo translação:
# - seta para esquerda: diminui a translação no eixo X
# - seta para direita: aumenta a translação no eixo X
# - seta para cima: diminui a translação no eixo Z
# - seta para baixo: aumenta a translação no eixo Z
# - espaço: aumenta a translação no eixo Y
# - ctrl: diminui a translação no eixo Y


import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
from pyrr import Matrix44, Vector3, matrix44, vector
import numpy as np
from PIL import Image
import math

# Vertex shader code - handles vertex positions, textures, normals and transformations
# Includes special handling for ground plane tiling and skybox sphere mapping
vertex_shader_code = """
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec2 aTexCoord;
layout(location = 2) in vec3 aNormal;

out vec2 TexCoord;
out vec3 FragPos;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform bool isGround;
uniform bool isSkybox;
uniform bool isSelected;

void main()
{
    if (isGround) {
        TexCoord = vec2(aTexCoord.x, 1.0 - aTexCoord.y) * 50.0; // Scale texture coordinates for ground
    } else if (isSkybox) {
        TexCoord = vec2(1.0 - aTexCoord.x, aTexCoord.y); // Invert x coordinate for skybox
    } else {
        TexCoord = vec2(aTexCoord.x, 1.0 - aTexCoord.y); // Normal texture coordinates for other objects
    }
    FragPos = vec3(model * vec4(aPos, 1.0));
    if (isSkybox) {
        vec3 spherePos = normalize(aPos) * 50.0; // Convert cube to sphere and scale
        gl_Position = projection * view * model * vec4(spherePos, 1.0);
    } else {
        gl_Position = projection * view * model * vec4(aPos, 1.0);
    }
}
"""

# Fragment shader code - handles texturing and transparency
fragment_shader_code = """
#version 330 core
out vec4 FragColor;

in vec2 TexCoord;
in vec3 FragPos;

uniform sampler2D texture_diffuse1;
uniform bool isGround;
uniform bool isSkybox;
uniform bool isSelected;

void main()
{
    vec4 texColor = texture(texture_diffuse1, TexCoord);
    if(texColor.a < 0.1) // Handle transparency
        discard;
    if(isSelected)
        FragColor = mix(texColor, vec4(1.0, 1.0, 0.0, 1.0), 0.3); // Yellow highlight for selected object
    else
        FragColor = texColor;
}
"""

def load_texture(path, is_ground=False):
    """
    Load and configure a texture from file
    Args:
        path: Path to texture file
        is_ground: Whether this is a ground texture that should repeat
    Returns:
        OpenGL texture ID
    """
    texture = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture)
    
    # Set texture parameters
    # If it's a ground texture, repeat it - avoid low texture quality
    if is_ground:
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    else:
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

    try:
        image = Image.open(path)
        img_data = np.array(image.convert("RGBA"))
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, img_data.shape[1], img_data.shape[0], 0, GL_RGBA, GL_UNSIGNED_BYTE, img_data)
        glGenerateMipmap(GL_TEXTURE_2D)
    except Exception as e:
        print(f"Error loading texture {path}: {e}")
        return 0
    
    return texture

def load_model(path):
    """
    Load 3D model from OBJ file
    Args:
        path: Path to OBJ file
    Returns:
        Numpy array of vertex data including positions, texture coords and normals
    """
    vertices, textures, normals, faces = [], [], [], []

    with open(path) as file:
        for line in file:
            parts = line.strip().split()
            if not parts: continue
            if parts[0] == 'v': vertices.append(list(map(float, parts[1:4])))
            elif parts[0] == 'vt': textures.append(list(map(float, parts[1:3])))
            elif parts[0] == 'vn': normals.append(list(map(float, parts[1:4])))
            elif parts[0] == 'f':
                face = []
                for vert in parts[1:]:
                    indices = [int(i)-1 if i else None for i in vert.split('/')]
                    face.append(indices + [None]*(3-len(indices)))
                faces.append(list(zip(*face)))

    vertex_data = []
    for face_verts, face_texs, face_norms in faces:
        vertices_to_process = face_verts[:3] + (face_verts[0], face_verts[2], face_verts[3]) if len(face_verts) == 4 else face_verts
        texs_to_process = face_texs[:3] + (face_texs[0], face_texs[2], face_texs[3]) if len(face_verts) == 4 else face_texs  
        norms_to_process = face_norms[:3] + (face_norms[0], face_norms[2], face_norms[3]) if len(face_verts) == 4 else face_norms

        for v, t, n in zip(vertices_to_process, texs_to_process, norms_to_process):
            vertex_data.extend(vertices[v] if v is not None and v < len(vertices) else [0.0, 0.0, 0.0])
            vertex_data.extend(textures[t] if t is not None and t < len(textures) else [0.0, 0.0])
            vertex_data.extend(normals[n] if n is not None and n < len(normals) else [0.0, 0.0, 0.0])

    return np.array(vertex_data, dtype=np.float32)

# Initial transformations for each object in the scene
transformations = {
    "cabin": {"translation": Vector3([0.0, -0.45, 0.0]), "scale": 0.5, "rotation": Vector3([0.0, 0.0, 0.0]), "scale_xyz": Vector3([1.5, 1.0, 1.0])},
    "rocks": {"translation": Vector3([5.0, 0.0, 10.0]), "scale": 0.05, "rotation": Vector3([0.0, 0.0, 90.0])},
    "table": {"translation": Vector3([-1.0, -0.35, -1.0]), "scale": 0.75, "rotation": Vector3([0.0, 0.0, 0.0])},
    "chair": {"translation": Vector3([-2.0, -0.35, -1.0]), "scale": 1.0, "rotation": Vector3([0.0, -90.0, 0.0])},
    "firepit": {"translation": Vector3([8.0, -0.35, 8.0]), "scale": 0.02, "rotation": Vector3([0.0, 0.0, 0.0])},
    "bed": {"translation": Vector3([0.0, -0.35, 1.1]), "scale": 0.7, "rotation": Vector3([0.0, 0.0, 0.0])},
    "dog": {"translation": Vector3([10.0, -0.4, 10.0]), "scale": 0.8, "rotation": Vector3([0.0, 135.0, 0.0])},
    "ground": {"translation": Vector3([0.0, -0.5, 0.0]), "scale": 50.0, "rotation": Vector3([0.0, 0.0, 0.0])},
    "skybox": {"translation": Vector3([0.0, 0.0, 0.0]), "scale": 1.0, "rotation": Vector3([0.0, 0.0, 0.0])}
}

# Object selection mapping
selectable_objects = {
    glfw.KEY_1: "cabin",
    glfw.KEY_2: "rocks", 
    glfw.KEY_3: "table",
    glfw.KEY_4: "chair",
    glfw.KEY_5: "firepit",
    glfw.KEY_6: "bed",
    glfw.KEY_7: "dog"
}

# Camera settings
camera_pos = Vector3([0.0, 2.0, 10.0])
camera_front = Vector3([0.0, 0.0, -1.0])
camera_up = Vector3([0.0, 1.0, 0.0])
yaw, pitch = -90.0, 0.0
first_mouse = True
last_x, last_y = 960, 540
fov = 45.0
modo_malha = False
selected_object = None
transform_mode = "rotation"  # Can be "rotation", "translation", or "scale"

def process_camera_input(window):
    """Handle camera movement with WASD keys"""
    global camera_pos, camera_front, camera_up
    global modo_malha, selected_object, transform_mode
    
    camera_speed = 0.05
    new_pos = camera_pos.copy()

    if glfw.get_key(window, glfw.KEY_W) == glfw.PRESS:
        new_pos += camera_speed * camera_front
    if glfw.get_key(window, glfw.KEY_S) == glfw.PRESS:
        new_pos -= camera_speed * camera_front
    if glfw.get_key(window, glfw.KEY_A) == glfw.PRESS:
        new_pos -= np.cross(camera_front, camera_up) * camera_speed
    if glfw.get_key(window, glfw.KEY_D) == glfw.PRESS:
        new_pos += np.cross(camera_front, camera_up) * camera_speed

    # Object transformations based on mode
    if selected_object:
        if transform_mode == "rotation":
            rotation_speed = 2.0
            if glfw.get_key(window, glfw.KEY_LEFT) == glfw.PRESS:
                transformations[selected_object]["rotation"].y -= rotation_speed
            if glfw.get_key(window, glfw.KEY_RIGHT) == glfw.PRESS:
                transformations[selected_object]["rotation"].y += rotation_speed
            if glfw.get_key(window, glfw.KEY_UP) == glfw.PRESS:
                transformations[selected_object]["rotation"].x -= rotation_speed
            if glfw.get_key(window, glfw.KEY_DOWN) == glfw.PRESS:
                transformations[selected_object]["rotation"].x += rotation_speed
        elif transform_mode == "translation":
            translation_speed = 0.1
            if glfw.get_key(window, glfw.KEY_LEFT) == glfw.PRESS:
                transformations[selected_object]["translation"].x -= translation_speed
            if glfw.get_key(window, glfw.KEY_RIGHT) == glfw.PRESS:
                transformations[selected_object]["translation"].x += translation_speed
            if glfw.get_key(window, glfw.KEY_UP) == glfw.PRESS:
                transformations[selected_object]["translation"].z -= translation_speed
            if glfw.get_key(window, glfw.KEY_DOWN) == glfw.PRESS:
                transformations[selected_object]["translation"].z += translation_speed
            if glfw.get_key(window, glfw.KEY_SPACE) == glfw.PRESS:
                transformations[selected_object]["translation"].y += translation_speed
            if glfw.get_key(window, glfw.KEY_LEFT_CONTROL) == glfw.PRESS:
                transformations[selected_object]["translation"].y -= translation_speed
        
        # Scaling with Z and X keys
        scale_speed = 0.01
        if glfw.get_key(window, glfw.KEY_Z) == glfw.PRESS:
            transformations[selected_object]["scale"] -= scale_speed
        if glfw.get_key(window, glfw.KEY_X) == glfw.PRESS:
            transformations[selected_object]["scale"] += scale_speed

    # Prevent camera from going below ground level and beyond skybox radius
    radius = 45  # Slightly smaller than skybox radius
    if new_pos.y >= 0.5 and (new_pos.x**2 + new_pos.y**2 + new_pos.z**2) <= radius**2:
        camera_pos = new_pos

def mouse_callback(window, xpos, ypos):
    """Handle mouse movement for camera rotation"""
    global yaw, pitch, last_x, last_y, first_mouse, camera_front

    if first_mouse:
        last_x, last_y = xpos, ypos
        first_mouse = False

    xoffset = xpos - last_x
    yoffset = last_y - ypos
    last_x, last_y = xpos, ypos

    sensitivity = 0.1
    xoffset *= sensitivity
    yoffset *= sensitivity

    yaw += xoffset
    pitch = max(-89.0, min(89.0, pitch + yoffset))

    # Calculate new camera direction
    front = Vector3([
        math.cos(math.radians(yaw)) * math.cos(math.radians(pitch)),
        math.sin(math.radians(pitch)),
        math.sin(math.radians(yaw)) * math.cos(math.radians(pitch))
    ])
    camera_front = vector.normalize(front)

def process_key_input(window, key, scancode, action, mods):
    """Handle keyboard input for toggling wireframe mode and object selection"""
    global modo_malha, selected_object, transform_mode
    if action == glfw.PRESS:
        if key == glfw.KEY_P:
            modo_malha = not modo_malha
        elif key in selectable_objects:
            selected_object = selectable_objects[key]
            print(f"Selected object: {selected_object}")
        elif key == glfw.KEY_R:
            transform_mode = "rotation"
            print("Rotation mode activated")
        elif key == glfw.KEY_T:
            transform_mode = "translation"
            print("Translation mode activated")

def generate_sphere_vertices(radius=1.0, sectors=1000, stacks=1000):
    """Generate vertices for a sphere (used for skybox)"""
    vertices = []
    
    for i in range(stacks + 1):
        V = i / stacks
        phi = V * math.pi
        
        for j in range(sectors + 1):
            U = j / sectors
            theta = U * 2 * math.pi
            
            x = math.cos(theta) * math.sin(phi)
            y = math.cos(phi)
            z = math.sin(theta) * math.sin(phi)
            
            vertices.extend([x, y, z])  # Position
            vertices.extend([U, V])     # Texture coordinates
            vertices.extend([x, y, z])  # Normal (same as position for sphere)
            
    indices = []
    for i in range(stacks):
        for j in range(sectors):
            first = i * (sectors + 1) + j
            second = first + sectors + 1
            
            indices.extend([first, second, first + 1])
            indices.extend([second, second + 1, first + 1])
    
    vertex_data = []
    for i in range(0, len(indices), 3):
        for j in range(3):
            idx = indices[i + j]
            base_idx = idx * 8
            vertex_data.extend(vertices[base_idx:base_idx + 8])
            
    return np.array(vertex_data, dtype=np.float32)

def main():
    """Main rendering function"""
    global camera_pos, camera_front, camera_up

    # Initialize GLFW and create window
    if not glfw.init():
        return
    window = glfw.create_window(1920, 1080, "Trabalho 2 - Computação Gráfica", None, None)
    if not window:
        glfw.terminate()
        return
    glfw.make_context_current(window)

    # Setup mouse input
    glfw.set_cursor_pos_callback(window, mouse_callback)
    glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_DISABLED)
    glfw.set_key_callback(window, process_key_input)

    glEnable(GL_DEPTH_TEST)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    # Compile shaders
    shader = compileProgram(compileShader(vertex_shader_code, GL_VERTEX_SHADER),
                          compileShader(fragment_shader_code, GL_FRAGMENT_SHADER))

    # Create ground plane vertices
    ground_vertices = np.array([
        # Position          # Texture coords  # Normals
        -1.0, 0.0, -1.0,   0.0, 0.0,         0.0, 1.0, 0.0,
         1.0, 0.0, -1.0,   1.0, 0.0,         0.0, 1.0, 0.0,
         1.0, 0.0,  1.0,   1.0, 1.0,         0.0, 1.0, 0.0,
        -1.0, 0.0, -1.0,   0.0, 0.0,         0.0, 1.0, 0.0,
         1.0, 0.0,  1.0,   1.0, 1.0,         0.0, 1.0, 0.0,
        -1.0, 0.0,  1.0,   0.0, 1.0,         0.0, 1.0, 0.0,
    ], dtype=np.float32)

    # Load models
    rock_model = load_model("objects/rock/rock.obj")
    models = {
        "cabin": load_model("objects/house/house.obj"),
        "rocks": rock_model,
        "table": load_model("objects/mesa/mesa.obj"),
        "chair": load_model("objects/cadeira/cadeira.obj"),
        "firepit": load_model("objects/campfire/Campfire_clean.OBJ"),
        "bed": load_model("objects/cama/cama.obj"),
        "dog": load_model("objects/dog/dog.obj"),
        "ground": ground_vertices,
        "skybox": generate_sphere_vertices(1.0, 30, 30)  # Generate sphere vertices for skybox
    }

    # Load textures
    rock_texture = load_texture("objects/rock/rock_texture.png")
    textures = {
        "cabin": load_texture("objects/house/house_None_AlbedoTransparency.png"),
        "rocks": rock_texture,
        "table": load_texture("objects/mesa/textura_mesa.jpg"),
        "chair": load_texture("objects/cadeira/textura_cadeira.jpg"),
        "firepit": load_texture("objects/campfire/Textures/HD/Campfire_MAT_BaseColor_01.jpg"),
        "bed": load_texture("objects/cama/textura_cama.jpg"),
        "dog": load_texture("objects/dog/Dog_Tris_Diffuse.png"),
        "ground": load_texture("objects/ground/sand-500-mm-architextures.jpg", True),
        "skybox": load_texture("objects/sky/free hdr_map_808.jpg")
    }

    # Create and setup VAOs/VBOs
    VAOs = glGenVertexArrays(len(models))
    VBOs = glGenBuffers(len(models))

    for i, (obj_name, data) in enumerate(models.items()):
        glBindVertexArray(VAOs[i])
        glBindBuffer(GL_ARRAY_BUFFER, VBOs[i])
        glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_STATIC_DRAW)

        # Position attribute
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * 4, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        # Texture coordinate attribute
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 8 * 4, ctypes.c_void_p(3 * 4))
        glEnableVertexAttribArray(1)
        # Normal attribute
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 8 * 4, ctypes.c_void_p(5 * 4))
        glEnableVertexAttribArray(2)

    # Setup projection matrix
    projection = Matrix44.perspective_projection(45.0, 1920 / 1080, 0.1, 100.0)
    glUseProgram(shader)
    glUniformMatrix4fv(glGetUniformLocation(shader, "projection"), 1, GL_FALSE, projection)
    glUniform3f(glGetUniformLocation(shader, "lightPos"), 5.0, 5.0, 5.0)

    # Main render loop
    while not glfw.window_should_close(window):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        process_camera_input(window)
        if modo_malha:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        else:
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
            
        # Update view matrix
        view = Matrix44.look_at(camera_pos, camera_pos + camera_front, camera_up)
        glUniformMatrix4fv(glGetUniformLocation(shader, "view"), 1, GL_FALSE, view)
        glUniform3f(glGetUniformLocation(shader, "viewPos"), camera_pos.x, camera_pos.y, camera_pos.z)
        
        # Render each object
        for i, obj_name in enumerate(models.keys()):
            trans = transformations[obj_name]
            scale_factor = 1.0

            # Calculate model matrix with separate x, y, z scaling for cabin
            if obj_name == "cabin":
                scale_xyz = trans.get("scale_xyz", Vector3([1.0, 1.0, 1.0]))
                model = (Matrix44.from_translation(trans["translation"]) *
                        Matrix44.from_scale(Vector3([trans["scale"] * scale_factor * scale_xyz.x,
                                                   trans["scale"] * scale_factor * scale_xyz.y,
                                                   trans["scale"] * scale_factor * scale_xyz.z])) *
                        Matrix44.from_x_rotation(np.radians(trans["rotation"].x)) *
                        Matrix44.from_y_rotation(np.radians(trans["rotation"].y)) *
                        Matrix44.from_z_rotation(np.radians(trans["rotation"].z)))
            else:
                model = (Matrix44.from_translation(trans["translation"]) *
                        Matrix44.from_scale(Vector3([trans["scale"] * scale_factor] * 3)) *
                        Matrix44.from_x_rotation(np.radians(trans["rotation"].x)) *
                        Matrix44.from_y_rotation(np.radians(trans["rotation"].y)) *
                        Matrix44.from_z_rotation(np.radians(trans["rotation"].z)))
            
            glUniformMatrix4fv(glGetUniformLocation(shader, "model"), 1, GL_FALSE, model)

            # Set if this is the ground object or skybox
            glUniform1i(glGetUniformLocation(shader, "isGround"), 1 if obj_name == "ground" else 0)
            glUniform1i(glGetUniformLocation(shader, "isSkybox"), 1 if obj_name == "skybox" else 0)
            glUniform1i(glGetUniformLocation(shader, "isSelected"), 1 if obj_name == selected_object else 0)

            # Bind VAO and texture
            glBindVertexArray(VAOs[i])
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, textures[obj_name])
            glUniform1i(glGetUniformLocation(shader, "texture_diffuse1"), 0)

            # Draw object
            glDrawArrays(GL_TRIANGLES, 0, len(models[obj_name]) // 8)

        glfw.swap_buffers(window)
        glfw.poll_events()

    glfw.terminate()

if __name__ == "__main__":
    main()
