# Trabalho 2 - Computação Gráfica
# Alunos: Hiago Vinicius Americo - 11218469, Vítor Beneti Martins - 11877635

# A cena representa o sentimento de solidão, com uma casa isolada no meio do deserto.
# O vazio da cena traz uma sensação de solidão e isolamento, enquanto o cachorro solitário passa a sensação de abandono.


# Controles: WASD - Movimentos da câmera
# P - ativa/desativa o modo malha


import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
from pyrr import Matrix44, Vector3, matrix44, vector
import numpy as np
from PIL import Image
import math
import cv2

# Material properties for each object
materials = {
    "cabin": {
        "diffuse": Vector3([0.8, 0.8, 0.8]),  # Wood-like diffuse
        "specular": Vector3([0.2, 0.2, 0.2])  # Low specular for wood
    },
    "rocks": {
        "diffuse": Vector3([0.6, 0.6, 0.6]),  # Stone-like diffuse
        "specular": Vector3([0.3, 0.3, 0.3])  # Medium specular for stone
    },
    "table": {
        "diffuse": Vector3([0.7, 0.7, 0.7]),  # Wood-like diffuse
        "specular": Vector3([0.1, 0.1, 0.1])  # Very low specular for wood
    },
    "chair": {
        "diffuse": Vector3([0.7, 0.7, 0.7]),  # Wood-like diffuse
        "specular": Vector3([0.1, 0.1, 0.1])  # Very low specular for wood
    },
    "firepit": {
        "diffuse": Vector3([0.9, 0.9, 0.9]),  # High diffuse for metal
        "specular": Vector3([1.0, 1.0, 1.0])  # High specular for metal
    },
    "bed": {
        "diffuse": Vector3([0.8, 0.8, 0.8]),  # Fabric-like diffuse
        "specular": Vector3([0.1, 0.1, 0.1])  # Low specular for fabric
    },
    "dog": {
        "diffuse": Vector3([0.8, 0.8, 0.8]),  # Fur-like diffuse
        "specular": Vector3([0.1, 0.1, 0.1])  # Low specular for fur
    },
    "ground": {
        "diffuse": Vector3([0.8, 0.8, 0.8]),  # Sand-like diffuse
        "specular": Vector3([0.0, 0.0, 0.0])  # No specular for sand
    },
    "skybox": {
        "diffuse": Vector3([1.0, 1.0, 1.0]),  # Full diffuse for sky
        "specular": Vector3([0.0, 0.0, 0.0])  # No specular for sky
    },
    "cactus": {
        "diffuse": Vector3([0.7, 0.8, 0.7]),  # Plant-like diffuse
        "specular": Vector3([0.2, 0.2, 0.2])  # Low specular for plant
    },
    "flashlight": {
        "diffuse": Vector3([0.9, 0.9, 0.9]),  # Metal-like diffuse
        "specular": Vector3([0.8, 0.8, 0.8])  # High specular for metal
    }
}

# Vertex shader code - handles vertex positions, textures, normals and transformations
# Includes special handling for ground plane tiling and skybox sphere mapping
vertex_shader_code = """
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec2 aTexCoord;
layout(location = 2) in vec3 aNormal;

out vec2 TexCoord;
out vec3 FragPos;
out vec3 Normal;
out vec4 vertexColor;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform bool isGround;
uniform bool isSkybox;
uniform bool isFirepit;

void main()
{
    if (isGround) {
        TexCoord = vec2(aTexCoord.x, 1.0 - aTexCoord.y) * 50.0;
    } else if (isSkybox) {
        TexCoord = vec2(1.0 - aTexCoord.x, aTexCoord.y);
    } else {
        TexCoord = vec2(aTexCoord.x, 1.0 - aTexCoord.y);
    }
    
    FragPos = vec3(model * vec4(aPos, 1.0));
    Normal = mat3(transpose(inverse(model))) * aNormal;
    
    if (isSkybox) {
        vec3 spherePos = normalize(aPos) * 50.0;
        gl_Position = projection * view * model * vec4(spherePos, 1.0);
    } else {
        gl_Position = projection * view * model * vec4(aPos, 1.0);
    }

    // Add emission color for firepit
    if (isFirepit) {
        vertexColor = vec4(1.5, 0.7, 0.3, 1.0); // Warm orange glow
    } else {
        vertexColor = vec4(1.0);
    }
}
"""

# Fragment shader code - handles texturing, lighting and transparency
fragment_shader_code = """
#version 330 core
out vec4 FragColor;

in vec2 TexCoord;
in vec3 FragPos;
in vec3 Normal;
in vec4 vertexColor;

uniform sampler2D texture_diffuse1;
uniform sampler2D texture_spikes;
uniform bool isGround;
uniform bool isSkybox;
uniform bool isCactus;
uniform bool isFirepit;
uniform vec3 viewPos;
uniform vec3 lightPos;  // Campfire position
uniform vec3 lightColor;  // Campfire color
uniform vec3 materialDiffuse;  // Material diffuse color
uniform vec3 materialSpecular;  // Material specular color

void main()
{
    vec4 texColor;
    if (isCactus) {
        vec4 baseColor = texture(texture_diffuse1, TexCoord);
        vec4 spikesColor = texture(texture_spikes, TexCoord);
        texColor = mix(baseColor, spikesColor, spikesColor.a);
    } else {
        texColor = texture(texture_diffuse1, TexCoord);
    }
    
    if(texColor.a < 0.1)
        discard;
        
    if(isSkybox) {
        FragColor = texColor;
    } else {
        // Ambient light (very low for night scene)
        float ambientStrength = 0.8;
        vec3 ambient = ambientStrength * vec3(0.05, 0.05, 0.1); // Bluish night ambient
        
        // Diffuse light from campfire using material properties
        vec3 norm = normalize(Normal);
        vec3 lightDir = normalize(lightPos - FragPos);
        float diff = max(dot(norm, lightDir), 0.0);
        float distance = length(lightPos - FragPos);
        float attenuation = 1.0 / (1.0 + 0.045 * distance + 0.0075 * distance * distance);
        vec3 diffuse = diff * lightColor * materialDiffuse * attenuation * 2.0;
        
        // Specular light using material properties
        vec3 viewDir = normalize(viewPos - FragPos);
        vec3 reflectDir = reflect(-lightDir, norm);
        float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
        vec3 specular = spec * lightColor * materialSpecular * attenuation;
        
        vec3 result = (ambient + diffuse + specular) * texColor.rgb;

        // Add emission for firepit
        if (isFirepit) {
            result += texColor.rgb * vertexColor.rgb * 0.5; // Add glow effect
        }

        FragColor = vec4(result, texColor.a);
    }
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
        if path.endswith('.hdr'):
            # Load HDR image using OpenCV
            img_data = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
            if img_data is None:
                raise Exception("Failed to load HDR image")
            # Convert to RGB
            img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
            # Normalize and convert to 8-bit
            img_data = np.clip(img_data * 255, 0, 255).astype(np.uint8)
            # Add alpha channel
            alpha = np.full((img_data.shape[0], img_data.shape[1], 1), 255, dtype=np.uint8)
            img_data = np.concatenate([img_data, alpha], axis=2)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, img_data.shape[1], img_data.shape[0], 0, GL_RGBA, GL_UNSIGNED_BYTE, img_data)
        else:
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
    "firepit": {"translation": Vector3([8.0, -0.45, 8.0]), "scale": 0.02, "rotation": Vector3([0.0, 0.0, 0.0])},
    "bed": {"translation": Vector3([-1.0, -0.35, 1.2]), "scale": 0.25, "rotation": Vector3([0.0, -90.0, 0.0])},
    "dog": {"translation": Vector3([10.0, -0.4, 10.0]), "scale": 0.8, "rotation": Vector3([0.0, 135.0, 0.0])},
    "ground": {"translation": Vector3([0.0, -0.5, 0.0]), "scale": 50.0, "rotation": Vector3([0.0, 0.0, 0.0])},
    "skybox": {"translation": Vector3([0.0, 0.0, 0.0]), "scale": 1.0, "rotation": Vector3([0.0, 0.0, 0.0])},
    "cactus": {"translation": Vector3([1.0, -1.0, 5.0]), "scale": 0.1, "rotation": Vector3([0.0, 45.0, 0.0])},
    "flashlight": {"translation": Vector3([-1.5, 0.2, -1.0]), "scale": 0.02, "rotation": Vector3([0.0, 45.0, 0.0])}
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

def process_camera_input(window):
    """Handle camera movement with WASD keys"""
    global camera_pos, camera_front, camera_up
    global modo_malha
    
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
    """Handle keyboard input for toggling wireframe mode"""
    global modo_malha
    if action == glfw.PRESS:
        if key == glfw.KEY_P:
            modo_malha = not modo_malha

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
        "bed": load_model("objects/bed/sleeping_bag.obj"),
        "dog": load_model("objects/dog/dog.obj"),
        "ground": ground_vertices,
        "skybox": generate_sphere_vertices(1.0, 30, 30),  # Generate sphere vertices for skybox
        "cactus": load_model("objects/cactus/cactus.obj"),
        "flashlight": load_model("objects/flashlight/flashlight.obj")
    }

    # Load textures
    rock_texture = load_texture("objects/rock/rock_texture.png")
    textures = {
        "cabin": load_texture("objects/house/house_None_AlbedoTransparency.png"),
        "rocks": rock_texture,
        "table": load_texture("objects/mesa/textura_mesa.jpg"),
        "chair": load_texture("objects/cadeira/textura_cadeira.jpg"),
        "firepit": load_texture("objects/campfire/Textures/HD/Campfire_MAT_BaseColor_01.jpg"),
        "bed": load_texture("objects/bed/SleepingBagDiffuse.png"),
        "dog": load_texture("objects/dog/Dog_Tris_Diffuse.png"),
        "ground": load_texture("objects/ground/sand-500-mm-architextures.jpg", True),
        "skybox": load_texture("objects/sky/clear_night_4k.hdr"),
        "cactus": load_texture("objects/cactus/diffuse.png"),
        "flashlight": load_texture("objects/flashlight/torch_BaseColor.png")
    }
    
    # Load cactus spikes texture
    cactus_spikes_texture = load_texture("objects/cactus/spikes.png")

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

    # Campfire light properties - stronger orange light
    campfire_pos = transformations["firepit"]["translation"]
    campfire_color = Vector3([1.5, 0.7, 0.3])  # Brighter warm orange color

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
        
        # Update lighting uniforms
        glUniform3f(glGetUniformLocation(shader, "viewPos"), camera_pos.x, camera_pos.y, camera_pos.z)
        glUniform3f(glGetUniformLocation(shader, "lightPos"), campfire_pos.x, campfire_pos.y, campfire_pos.z)
        glUniform3f(glGetUniformLocation(shader, "lightColor"), campfire_color.x, campfire_color.y, campfire_color.z)
        
        # Render each object
        for i, obj_name in enumerate(models.keys()):
            trans = transformations[obj_name]
            scale_factor = 1.0

            # Set material properties for current object
            material = materials[obj_name]
            glUniform3f(glGetUniformLocation(shader, "materialDiffuse"), 
                       material["diffuse"].x, material["diffuse"].y, material["diffuse"].z)
            glUniform3f(glGetUniformLocation(shader, "materialSpecular"),
                       material["specular"].x, material["specular"].y, material["specular"].z)

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

            # Set if this is the ground object, skybox, cactus or firepit
            glUniform1i(glGetUniformLocation(shader, "isGround"), 1 if obj_name == "ground" else 0)
            glUniform1i(glGetUniformLocation(shader, "isSkybox"), 1 if obj_name == "skybox" else 0)
            glUniform1i(glGetUniformLocation(shader, "isCactus"), 1 if obj_name == "cactus" else 0)
            glUniform1i(glGetUniformLocation(shader, "isFirepit"), 1 if obj_name == "firepit" else 0)

            # Bind VAO and textures
            glBindVertexArray(VAOs[i])
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, textures[obj_name])
            glUniform1i(glGetUniformLocation(shader, "texture_diffuse1"), 0)
            
            # Bind spikes texture for cactus
            if obj_name == "cactus":
                glActiveTexture(GL_TEXTURE1)
                glBindTexture(GL_TEXTURE_2D, cactus_spikes_texture)
                glUniform1i(glGetUniformLocation(shader, "texture_spikes"), 1)

            # Draw object
            glDrawArrays(GL_TRIANGLES, 0, len(models[obj_name]) // 8)

        glfw.swap_buffers(window)
        glfw.poll_events()

    glfw.terminate()

if __name__ == "__main__":
    main()
