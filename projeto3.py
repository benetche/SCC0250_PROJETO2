# Trabalho 3 - Computação Gráfica
# Alunos: Hiago Vinicius Americo - 11218469, Vítor Beneti Martins - 11877635

# A cena representa o sentimento de solidão, com uma casa isolada no meio do deserto.
# O vazio da cena traz uma sensação de solidão e isolamento, enquanto o cachorro solitário passa a sensação de abandono.


# Controles: WASD - Movimentos da câmera
# P - ativa/desativa o modo malha
# + - aumenta a intensidade da luz ambiente
# - diminui a intensidade da luz ambiente
# I - aumenta a reflexão difusa
# O - diminui a reflexão difusa
# K - aumenta a reflexão especular
# L - diminui a reflexão especular


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
    "flashlight": {
        "diffuse": Vector3([0.9, 0.9, 0.9]),  # Metal-like diffuse
        "specular": Vector3([0.8, 0.8, 0.8])  # High specular for metal
    },
    "lantern": {
        "diffuse": Vector3([0.9, 0.9, 0.9]),  # Metal-like diffuse
        "specular": Vector3([0.8, 0.8, 0.8])  # High specular for metal
    },
    "lantern_glow": {
        "diffuse": Vector3([1.0, 1.0, 0.0]),  # Yellow glow
        "specular": Vector3([1.0, 1.0, 0.0])  # Yellow specular
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
uniform bool isLanternGlow;
uniform bool isInsideCabin; // New uniform to check if inside cabin

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

    // Add emission color for firepit and lantern glow
    if (isFirepit) {
        vertexColor = vec4(1.5, 0.7, 0.3, 1.0); // Warm orange glow
    } else if (isLanternGlow) {
        vertexColor = vec4(1.0, 1.0, 0.0, 0.3); // Yellow glow with transparency
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
uniform bool isGround;
uniform bool isSkybox;
uniform bool isFirepit;
uniform bool isLanternGlow;
uniform bool isInsideCabin; // New uniform to check if inside cabin
uniform vec3 viewPos;
uniform vec3 lightPos;  // Campfire position
uniform vec3 lightColor;  // Campfire color
uniform vec3 materialDiffuse;  // Material diffuse color
uniform vec3 materialSpecular;  // Material specular color
uniform float ambientStrength;  // Ambient light strength

// Flashlight uniforms
uniform vec3 flashlightPos;      // Flashlight position
uniform vec3 flashlightDir;      // Flashlight beam direction
uniform vec3 flashlightColor;    // Flashlight light color
uniform float flashlightCutOff;  // Flashlight cone angle (cosine)
uniform float flashlightOuterCutOff; // Outer angle of the cone (cosine)

// Lantern light properties
uniform vec3 lanternLightPos;    // Lantern light position
uniform vec3 lanternLightColor;  // Lantern light color

void main()
{
    vec4 texColor = texture(texture_diffuse1, TexCoord);
    
    if(texColor.a < 0.1 && !isLanternGlow)
        discard;
        
    if(isSkybox) {
        FragColor = texColor;
    } else if(isLanternGlow) {
        FragColor = vertexColor;
    } else {
        // Ambient light (very low for night scene)
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
        
        // Flashlight lighting calculation
        vec3 flashlightEffect = vec3(0.0);
        vec3 flashlightLightDir = normalize(flashlightPos - FragPos);
        float theta = dot(flashlightLightDir, normalize(-flashlightDir));

        if(theta > flashlightOuterCutOff) {
            // Inside the cone
            float epsilon = flashlightCutOff - flashlightOuterCutOff;
            float intensity = clamp((theta - flashlightOuterCutOff) / epsilon, 0.0, 1.0);
            
            // Distance-based attenuation
            float flashlightDistance = length(flashlightPos - FragPos);
            float flashlightAttenuation = 1.0 / (1.0 + 0.045 * flashlightDistance + 0.0075 * flashlightDistance * flashlightDistance);
            
            vec3 flashlightDiffuse = intensity * flashlightColor * materialDiffuse * flashlightAttenuation;
            vec3 flashlightSpecular = intensity * flashlightColor * materialSpecular * flashlightAttenuation;
            flashlightEffect = flashlightDiffuse + flashlightSpecular;
        }

        // Lantern lighting calculation
        vec3 lanternEffect = vec3(0.0);
        vec3 lanternLightDir = normalize(lanternLightPos - FragPos);
        float lanternDistance = length(lanternLightPos - FragPos);
        float lanternAttenuation = 1.0 / (1.0 + 0.045 * lanternDistance + 0.0075 * lanternDistance * lanternDistance);
        float lanternDiff = max(dot(norm, lanternLightDir), 0.0);
        vec3 lanternDiffuse = lanternDiff * lanternLightColor * materialDiffuse * lanternAttenuation;
        vec3 lanternSpecular = spec * lanternLightColor * materialSpecular * lanternAttenuation;
        lanternEffect = lanternDiffuse + lanternSpecular;

        vec3 result = (ambient + diffuse + specular + flashlightEffect + lanternEffect) * texColor.rgb;

        // Add emission for firepit
        if (isFirepit) {
            result += texColor.rgb * vertexColor.rgb * 0.5; // Add glow effect
        }

        // If inside the cabin, only use cabin light
        if (isInsideCabin) {
            result = ambient * texColor.rgb; // Only ambient light for cabin objects
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
    "cabin": {"translation": Vector3([0.0, -0.45, 0.0]), "scale": 0.5, "rotation": Vector3([0.0, -90.0, 0.0]), "scale_xyz": Vector3([1.5, 1.0, 1.0])},
    "rocks": {"translation": Vector3([5.0, 0.0, 10.0]), "scale": 0.05, "rotation": Vector3([0.0, 0.0, 90.0])},
    "table": {"translation": Vector3([-1.0, -0.35, -1.0]), "scale": 0.75, "rotation": Vector3([0.0, 0.0, 0.0])},
    "chair": {"translation": Vector3([-2.0, -0.35, -1.0]), "scale": 1.0, "rotation": Vector3([0.0, -90.0, 0.0])},
    "firepit": {"translation": Vector3([10.0, -0.45, 15.0]), "scale": 0.02, "rotation": Vector3([0.0, 0.0, 0.0])},
    "bed": {"translation": Vector3([-1.0, -0.35, 1.2]), "scale": 0.25, "rotation": Vector3([0.0, -90.0, 0.0])},
    "dog": {"translation": Vector3([10.0, -0.4, 10.0]), "scale": 0.8, "rotation": Vector3([0.0, 135.0, 0.0])},
    "ground": {"translation": Vector3([0.0, -0.5, 0.0]), "scale": 50.0, "rotation": Vector3([0.0, 0.0, 0.0])},
    "skybox": {"translation": Vector3([0.0, 0.0, 0.0]), "scale": 1.0, "rotation": Vector3([0.0, 0.0, 0.0])},
    "flashlight": {"translation": Vector3([-0.92, 0.2, -0.75]), "scale": 0.02, "rotation": Vector3([0.0, -95.0, 0.0])},
    "lantern": {"translation": Vector3([2.0, -0.2, 0.0]), "scale": 0.1, "rotation": Vector3([0.0, 0.0, 0.0])},
    "lantern_glow": {"translation": Vector3([2.0, -0.2, 0.0]), "scale": 0.15, "rotation": Vector3([0.0, 0.0, 0.0])}  # Slightly larger scale for glow
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
ambient_strength = 0.2  # Initial ambient light strength
flashlight_on = True    # Initial state of the flashlight
lantern_on = True      # Initial state of the lantern
campfire_on = True     # Initial state of the campfire
ambient_on = True      # Initial state of ambient light

diffuse_strength = 1.0  # Initial diffuse reflection strength
specular_strength = 1.0  # Initial specular reflection strength

def process_campfire_input(window):
    """Handle campfire movement with arrow keys"""
    global transformations
    
    translation_speed = 0.1
    campfire = transformations["firepit"]
    
    if glfw.get_key(window, glfw.KEY_LEFT) == glfw.PRESS:
        campfire["translation"].x -= translation_speed
    if glfw.get_key(window, glfw.KEY_RIGHT) == glfw.PRESS:
        campfire["translation"].x += translation_speed
    if glfw.get_key(window, glfw.KEY_UP) == glfw.PRESS:
        campfire["translation"].z -= translation_speed
    if glfw.get_key(window, glfw.KEY_DOWN) == glfw.PRESS:
        campfire["translation"].z += translation_speed

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
    """Handle keyboard input for toggling wireframe mode and adjusting ambient light"""
    global modo_malha, ambient_strength, diffuse_strength, specular_strength
    global flashlight_on, lantern_on, campfire_on, ambient_on
    if action == glfw.PRESS:
        if key == glfw.KEY_P:
            modo_malha = not modo_malha
        elif key == glfw.KEY_KP_ADD or key == glfw.KEY_EQUAL:
            ambient_strength = min(ambient_strength + 0.1, 1.0)
        elif key == glfw.KEY_KP_SUBTRACT or key == glfw.KEY_MINUS:
            ambient_strength = max(ambient_strength - 0.1, 0.0)
        elif key == glfw.KEY_I:
            diffuse_strength = min(diffuse_strength + 0.1, 2.0)
        elif key == glfw.KEY_O:
            diffuse_strength = max(diffuse_strength - 0.1, 0.0)
        elif key == glfw.KEY_K:
            specular_strength = min(specular_strength + 0.1, 2.0)
        elif key == glfw.KEY_L:
            lantern_on = not lantern_on
        elif key == glfw.KEY_F:
            flashlight_on = not flashlight_on
        elif key == glfw.KEY_C:
            campfire_on = not campfire_on
        elif key == glfw.KEY_Z:
            ambient_on = not ambient_on

            
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
    global camera_pos, camera_front, camera_up, ambient_strength, diffuse_strength, specular_strength

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

    # Create cube vertices for lantern glow
    lantern_glow_vertices = np.array([
        # Front face
        -0.65, -0.65,  0.65,   0.0, 0.0,   0.0, 0.0, 1.0,
         0.65, -0.65,  0.65,   1.0, 0.0,   0.0, 0.0, 1.0,
         0.65,  0.65,  0.65,   1.0, 1.0,   0.0, 0.0, 1.0,
        -0.65, -0.65,  0.65,   0.0, 0.0,   0.0, 0.0, 1.0,
         0.65,  0.65,  0.65,   1.0, 1.0,   0.0, 0.0, 1.0,
        -0.65,  0.65,  0.65,   0.0, 1.0,   0.0, 0.0, 1.0,
        # Back face
        -0.65, -0.65, -0.65,   0.0, 0.0,   0.0, 0.0, -1.0,
         0.65, -0.65, -0.65,   1.0, 0.0,   0.0, 0.0, -1.0,
         0.65,  0.65, -0.65,   1.0, 1.0,   0.0, 0.0, -1.0,
        -0.65, -0.65, -0.65,   0.0, 0.0,   0.0, 0.0, -1.0,
         0.65,  0.65, -0.65,   1.0, 1.0,   0.0, 0.0, -1.0,
        -0.65,  0.65, -0.65,   0.0, 1.0,   0.0, 0.0, -1.0,
        # Top face
        -0.65,  0.65, -0.65,   0.0, 0.0,   0.0, 1.0, 0.0,
         0.65,  0.65, -0.65,   1.0, 0.0,   0.0, 1.0, 0.0,
         0.65,  0.65,  0.65,   1.0, 1.0,   0.0, 1.0, 0.0,
        -0.65,  0.65, -0.65,   0.0, 0.0,   0.0, 1.0, 0.0,
         0.65,  0.65,  0.65,   1.0, 1.0,   0.0, 1.0, 0.0,
        -0.65,  0.65,  0.65,   0.0, 1.0,   0.0, 1.0, 0.0,
        # Bottom face
        -0.65, -0.65, -0.65,   0.0, 0.0,   0.0, -1.0, 0.0,
         0.65, -0.65, -0.65,   1.0, 0.0,   0.0, -1.0, 0.0,
         0.65, -0.65,  0.65,   1.0, 1.0,   0.0, -1.0, 0.0,
        -0.65, -0.65, -0.65,   0.0, 0.0,   0.0, -1.0, 0.0,
         0.65, -0.65,  0.65,   1.0, 1.0,   0.0, -1.0, 0.0,
        -0.65, -0.65,  0.65,   0.0, 1.0,   0.0, -1.0, 0.0,
        # Right face
         0.65, -0.65, -0.65,   0.0, 0.0,   1.0, 0.0, 0.0,
         0.65,  0.65, -0.65,   1.0, 0.0,   1.0, 0.0, 0.0,
         0.65,  0.65,  0.65,   1.0, 1.0,   1.0, 0.0, 0.0,
         0.65, -0.65, -0.65,   0.0, 0.0,   1.0, 0.0, 0.0,
        # Left face
        -0.65, -0.65, -0.65,   0.0, 0.0,   -1.0, 0.0, 0.0,
        -0.65,  0.65, -0.65,   1.0, 0.0,   -1.0, 0.0, 0.0,
        -0.65,  0.65,  0.65,   1.0, 1.0,   -1.0, 0.0, 0.0,
        -0.65, -0.65, -0.65,   0.0, 0.0,   -1.0, 0.0, 0.0,
         0.65, -0.65,  0.65,   1.0, 1.0,   1.0, 0.0, 0.0,
        -0.65,  0.65,  0.65,   0.0, 1.0,   -1.0, 0.0, 0.0,
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
        "flashlight": load_model("objects/flashlight/flashlight.obj"),
        "lantern": load_model("objects/lantern/lantern.obj"),
        "lantern_glow": lantern_glow_vertices  # Add lantern glow vertices
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
        "flashlight": load_texture("objects/flashlight/torch_BaseColor.png"),
        "lantern": load_texture("objects/lantern/lantern_base.png"),
        "lantern_glow": 0  # No texture for lantern glow
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

    # Campfire light properties - stronger orange light
    campfire_pos = transformations["firepit"]["translation"]
    campfire_color = Vector3([1.5, 0.7, 0.3])  # Brighter warm orange color

    # Flashlight position and direction
    flashlight_position = np.array([-0.92, 0.4, -0.6], dtype=np.float32)  # Adjust as needed
    flashlight_direction = np.array([0.2, 0.0, 2.0], dtype=np.float32)  # Light direction (normalized vector) - rotated by 180 degrees
    flashlight_color = np.array([1.0, 1.0, 1.0], dtype=np.float32)  # White light
    flashlight_cutoff = np.cos(np.radians(12.5))  # Inner cone angle
    flashlight_outer_cutoff = np.cos(np.radians(17.5))  # Outer cone angle

    # Lantern light properties
    lantern_light_pos = transformations["lantern"]["translation"]
    lantern_light_color = Vector3([1.0, 1.0, 0.0])  # Yellow light

    # Main render loop
    while not glfw.window_should_close(window):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        process_camera_input(window)
        process_campfire_input(window)  

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
        # Update ambient light
        if ambient_on:
            glUniform1f(glGetUniformLocation(shader, "ambientStrength"), ambient_strength)
        else:
            glUniform1f(glGetUniformLocation(shader, "ambientStrength"), 0.0)
        
        # Update flashlight uniforms
        # Update flashlight uniforms
        
        # Update campfire light uniforms
        glUniform3f(glGetUniformLocation(shader, "lightPos"), campfire_pos.x, campfire_pos.y, campfire_pos.z)
        if campfire_on:
            glUniform3f(glGetUniformLocation(shader, "lightColor"), campfire_color.x, campfire_color.y, campfire_color.z)
        else:
            glUniform3f(glGetUniformLocation(shader, "lightColor"), 0.0, 0.0, 0.0)

        # Update flashlight uniforms
        glUniform3fv(glGetUniformLocation(shader, "flashlightPos"), 1, flashlight_position)
        glUniform3fv(glGetUniformLocation(shader, "flashlightDir"), 1, flashlight_direction)
        if flashlight_on:
            glUniform3fv(glGetUniformLocation(shader, "flashlightColor"), 1, flashlight_color)
        else:
            glUniform3fv(glGetUniformLocation(shader, "flashlightColor"), 1, np.array([0.0, 0.0, 0.0], dtype=np.float32))
        glUniform1f(glGetUniformLocation(shader, "flashlightCutOff"), flashlight_cutoff)
        glUniform1f(glGetUniformLocation(shader, "flashlightOuterCutOff"), flashlight_outer_cutoff)

        # Update lantern light uniforms
        glUniform3f(glGetUniformLocation(shader, "lanternLightPos"), lantern_light_pos.x, lantern_light_pos.y, lantern_light_pos.z)
        if lantern_on:
            glUniform3f(glGetUniformLocation(shader, "lanternLightColor"), lantern_light_color.x, lantern_light_color.y, lantern_light_color.z)
        else:
            glUniform3f(glGetUniformLocation(shader, "lanternLightColor"), 0.0, 0.0, 0.0)
        # Render each object
        for i, obj_name in enumerate(models.keys()):
            trans = transformations[obj_name]
            scale_factor = 1.0

            # Set material properties for current object
            material = materials[obj_name]
            glUniform3f(glGetUniformLocation(shader, "materialDiffuse"), 
                       material["diffuse"].x * diffuse_strength, material["diffuse"].y * diffuse_strength, material["diffuse"].z * diffuse_strength)
            glUniform3f(glGetUniformLocation(shader, "materialSpecular"),
                       material["specular"].x * specular_strength, material["specular"].y * specular_strength, material["specular"].z * specular_strength)

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

            # Set if this is the ground object, skybox, firepit, or lantern glow
            glUniform1i(glGetUniformLocation(shader, "isGround"), 1 if obj_name == "ground" else 0)
            glUniform1i(glGetUniformLocation(shader, "isSkybox"), 1 if obj_name == "skybox" else 0)
            glUniform1i(glGetUniformLocation(shader, "isFirepit"), 1 if obj_name == "firepit" else 0)
            glUniform1i(glGetUniformLocation(shader, "isLanternGlow"), 1 if obj_name == "lantern_glow" else 0)

            # Bind VAO and textures
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
