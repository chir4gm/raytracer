import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import math
from typing import List

class Vec2:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def to_numpy(self):
        return np.array([self.x, self.y])

    def __add__(self, other):
        return Vec2(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Vec2(self.x - other.x, self.y - other.y)

    def __abs__(self):
        return math.sqrt(self.x**2 + self.y**2)

    def normalize(self):
        magnitude = abs(self)
        if magnitude != 0:
            self.x /= magnitude
            self.y /= magnitude

    def __neg__(self):
        return Vec2(-self.x, -self.y)

    def __mul__(self, other):
        if isinstance(other,Vec2):
            return self.x*other.x + self.y*other.y
        elif isinstance(other, (int, float, np.uint8)):
            return Vec2(self.x * other, self.y * other)
        else:
            raise ValueError(f"Unsupported operand type for *: 'Vec2' and '{type(other)}'")

class Vec3:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def to_numpy(self):
        return np.array([self.x, self.y, self.z])

    def __neg__(self):
        return Vec3(-self.x, -self.y, -self.z)

    def __add__(self, other):
        if isinstance(other,Vec3):
            return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)
        elif isinstance(other,np.ndarray):
            return Vec3(self.x + other[0], self.y + other[1], self.z + other[2])

    def __sub__(self, other):
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __abs__(self):
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def normalize(self):
        magnitude = abs(self)
        if magnitude != 0:
            self.x /= magnitude
            self.y /= magnitude
            self.z /= magnitude

    def __mul__(self, other):
        if isinstance(other,Vec3):
            return self.x*other.x + self.y*other.y + self.z*other.z
        elif isinstance(other,np.ndarray):
            return self.x*other[0] + self.y*other[1] + self.z*other[2]
        elif isinstance(other, (int, float, np.uint8)):
            return Vec3(self.x * other, self.y * other, self.z * other)
        else:
            raise ValueError(f"Unsupported operand type for *: 'Vec3' and '{type(other)}'")

def reflect(incident, normal):
    return incident - normal*2.0*(incident*normal)

def refract(incident, normal, refractive_index):
    cosine = max(-1.0, min(1, incident * normal))
    if cosine < 0:
        return refract(incident, -normal, refractive_index)
    k = 1 - (refractive_index**2)*(1-cosine**2)
    if k < 0:
        return Vec3(1.0,1.0,1.0)
    else:
        return incident * refractive_index + normal * (refractive_index * cosine - math.sqrt(k))

class Material:
    def __init__(self, diffuse_color: Vec3, spec_exp, refractive_index, albedo=(1.0, 0.0, 0.0)):
        self.diffuse_color = diffuse_color
        self.specular_exponent = spec_exp
        self.albedo = albedo
        self.refractive_index = refractive_index

class Checkboard:
    def __init__(self, plane_y, material_even, material_odd):
        self.plane_y = plane_y
        self.material_even = material_even
        self.material_odd = material_odd

    def intersect(self, ray_origin, ray_direction):
        if abs(ray_direction.y - 0) < 1e-5:
            if abs(ray_origin.y - self.plane_y) < 1e-5:
                return ray_origin
        t = (self.plane_y - ray_origin.y) / ray_direction.y
        if t > 0:
            intersection_point =  ray_origin + ray_direction * t
            if abs(intersection_point - ray_origin) < 300.0:
                return intersection_point
        return None
    def get_intersection_material(self, intersection_point):
        z = math.floor(0.5*intersection_point.z) 
        x = math.floor(0.5*intersection_point.x) 
        if (x + z) % 2:
            return self.material_even
        else:
            return self.material_odd
    def get_intersection_normal(self, intersection_normal):
        # always be pointin' up
        return Vec3(0.0, 1.0, 0.0) 

class Sphere:
    def __init__(self, center, radius, material : Material):
        self.center = center
        self.radius = radius
        self.material = material

    def intersect(self, ray_origin : Vec3, ray_direction : Vec3):
        intersection_point = None
        ray_direction.normalize()
        origin_to_center = self.center - ray_origin
        proj_length = ray_direction * origin_to_center
        if proj_length > 0:
            proj_point = ray_origin + ray_direction * proj_length
            if abs(proj_point - self.center) <= self.radius:
                distance = math.sqrt(self.radius ** 2 - abs(proj_point - self.center)**2) 
                di1 = None
                if abs(proj_length) > self.radius:
                    di1 = abs(proj_point - ray_origin) - distance
                else:
                    di1 = abs(proj_point - ray_origin) + distance
                intersection_point = ray_origin + ray_direction*di1
            else:
                return None
        else:
            if abs(origin_to_center) > self.radius:
                return None
            elif abs(origin_to_center) == self.radius:
                return ray_origin
            else:
                proj_point = ray_origin + ray_direction * proj_length
                distance = math.sqrt(self.radius ** 2 - abs(proj_point - self.center) ** 2)
                di1 = distance - abs(proj_point - ray_origin)
                intersection_point = ray_origin + ray_direction*di1
        return intersection_point 

    def get_intersection_material(self, intersection_point):
        return self.material
    
    def get_intersection_normal(self, point:Vec3):
        normal = point - self.center
        normal.normalize()
        return normal

class Light:
    def __init__(self, position: Vec3, intensity: float):
        self.position = position
        self.intensity = intensity

def write_ppm(image_matrix: np.ndarray, image_file):
    width = image_matrix.shape[1]
    height = image_matrix.shape[0]

    with open(image_file, 'w') as f:
        f.write("P3\n")
        f.write(f"{width} {height}\n")
        f.write(f"255\n")
        for i in range(height):
            for j in range(width):
                f.write(f"{image_matrix[i][j][0]} {image_matrix[i][j][1]} {image_matrix[i][j][2]}\n")
        f.close()

class Background:
    def __init__(self, file):
        self.background = Image.open(file)
    def get_color(self, direction:Vec3):
        direction.normalize()
        theta = math.atan2(direction.z, direction.x)
        phi = math.acos(direction.y)
        image_width, image_height = self.background.size
        u = ((theta + math.pi)/(2*math.pi)) * image_width
        v = (phi / math.pi) * image_height
        return np.array(self.background.getpixel((u,v)), dtype=np.uint8)

scene_background = Background("envmap.jpg")
ivory =         Material(diffuse_color=Vec3(0.4, 0.4, 0.3) * np.uint8(255), albedo=(0.9,0.5,0.1, 0.0), spec_exp=50, refractive_index=1.0)
glass =         Material(diffuse_color=Vec3(0.6, 0.7, 0.8) * np.uint8(255), albedo=(0.0,0.9,0.1, 0.8), spec_exp=125, refractive_index=1.5)
red_rubber =    Material(diffuse_color=Vec3(0.3, 0.1, 0.1) * np.uint8(255), albedo=(1.4,0.3,0.0,0.0), spec_exp=10, refractive_index=1.0)
mirror =        Material(diffuse_color=Vec3(1.0, 1.0, 1.0) * np.uint8(255), albedo=(0.0, 16.0, 0.8, 0.0), spec_exp=1425.0, refractive_index=1.0)

scene_objects = [Sphere(Vec3(-3,    0,   -16), 2,      ivory),
    Sphere(Vec3(-1.0, -1.5, -12), 2, glass),
    Sphere(Vec3( 1.5, -0.5, -18), 3, red_rubber),
    Sphere(Vec3( 7,    5,   -18), 4,      mirror),
    Checkboard(-10, red_rubber, ivory)]
# scene_objects = [Sphere(Vec3(1.0, 2.0, -5.0), 3, ivory), 
#                  Sphere(Vec3(-3.0, -4.0, -10.0), 5, red_rubber)]
lights = [Light(Vec3(0.0, 0.0, 0.0), 1.5),
          Light(Vec3( 30, 50, -25), 1.8),
          Light(Vec3( 30, 20,  30), 1.7)]

# Returns 
def scene_intersect(origin, direction, scene_objects):
    closest_object_idx = None
    closest_object_dist = np.inf
    closest_intersection_point = None
    for obj_idx in range(len(scene_objects)):
        obj = scene_objects[obj_idx]
        intersection_point = obj.intersect(origin, direction)
        if intersection_point is not None:
            if abs(intersection_point - origin) < closest_object_dist:
                closest_object_idx = obj_idx
                closest_object_dist = abs(intersection_point - origin)
                closest_intersection_point = intersection_point
    if closest_object_idx is not None:
        closest_object = scene_objects[closest_object_idx]
        return closest_intersection_point, closest_object 
    else:
        return None, -1

# Returns a color
def scene_cast_ray(origin, direction, scene_objects, lights, scene_background, depth=0) -> Vec3:
    intersection_point, closest_object = scene_intersect(origin, direction, scene_objects)
    if depth > 3 or intersection_point is None:
        return scene_background.get_color(direction)
    normal = closest_object.get_intersection_normal(intersection_point)
    closest_object_material = closest_object.get_intersection_material(intersection_point)
    reflection_dir = reflect(direction, normal)
    reflection_dir.normalize()

    refraction_dir = refract(direction, normal, closest_object_material.refractive_index)
    refraction_dir.normalize()
    
    reflection_color = scene_cast_ray(intersection_point, reflection_dir, scene_objects, lights, scene_background, depth + 1)
    refraction_color = scene_cast_ray(intersection_point, refraction_dir, scene_objects, lights, scene_background, depth + 1)

    view_direction = -direction
    diffuse_light_intensity = 0.0
    specular_light_intensity = 0.0

    for light in lights:
        light_dir = light.position - intersection_point 
        light_dist = abs(light_dir)
        light_dir.normalize()

        # If there is a sphere between the intersection point and light source, this point is in the shadow of the sphere
        # and the current light source
        bias = 1e-4
        shadow_origin = intersection_point
        if light_dir * normal > 0:
            shadow_origin += normal * bias 
        else:
            shadow_origin += normal * -bias
        shadow_point, _ = scene_intersect(shadow_origin, light_dir, scene_objects) 
        skip = False
        if shadow_point is not None:
            if abs(shadow_point - shadow_origin) < light_dist:
                skip = True
        if skip:
            continue
        diffuse_light_intensity += max(0, normal * light_dir) * light.intensity
        reflection_dir = reflect(-light_dir, normal)
        specular_light_intensity += light.intensity * max(0, reflection_dir * view_direction) ** closest_object_material.specular_exponent
    
    color = closest_object_material.diffuse_color * diffuse_light_intensity * closest_object_material.albedo[0]
    color += Vec3(255,255,255) * specular_light_intensity * closest_object_material.albedo[1]
    color += (reflection_color * closest_object_material.albedo[2])
    color += (refraction_color * closest_object_material.albedo[3])
    color = np.clip(color.to_numpy(), 0, 255).astype(np.uint8)
    return color

def render(scene_objects, lights : List[Light], cam_origin, cam_normal, scene_background, img_w, img_h):
    img_matrix = np.ndarray((img_h,img_w, 3), dtype=np.uint8)
    fov = 70.0
    fov = (fov/360) * (2*math.pi)
    screen_dist = 1.0
    up_dir = (math.tan(fov/2) * screen_dist) / (img_h / 2.0)
    right_dir = (math.tan(fov/2) * screen_dist) / (img_w / 2.0)
    pbar = tqdm(total=img_h*img_w)
    for i in range(img_h):
        for j in range(img_w):
            cam_direction = Vec3(right_dir * ((j + 0.5) - img_w/2),
                                 up_dir * (img_h/2 - (i + 0.5)), 0) + cam_normal
            cam_direction.normalize()
            img_matrix[i][j] = scene_cast_ray(cam_origin, cam_direction, scene_objects, lights, scene_background)
            pbar.update(1)
    pbar.close()

    return img_matrix

img_matrix = render(scene_objects, lights, Vec3(0.0,0.0,0.0), Vec3(0.0, 0.0, -1.0), scene_background, 1024, 768)
write_ppm(img_matrix, "img.ppm")