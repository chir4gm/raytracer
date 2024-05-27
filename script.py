import numpy as np
import matplotlib.pyplot as plt
import math
from typing import List


class Vec3:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def to_numpy(self):
        return np.array([self.x, self.y, self.z])
    def __add__(self, other):
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __abs__(self):
        return (self.x**2 + self.y**2 + self.z**2)**(1/2)

    def normalize(self):
        magnitude = abs(self)
        if magnitude != 0:
            self.x /= magnitude
            self.y /= magnitude
            self.z /= magnitude

    def __mul__(self, other):
        if isinstance(other,Vec3):
            return self.x*other.x + self.y*other.y + self.z*other.z
        else:
            return Vec3(self.x * other, self.y * other, self.z * other)

class Material:
    def __init__(self, color: Vec3):
        self.diffuse_color = color


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
    
    def normal_at(self, point:Vec3):
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
scene_background = np.array([55, 66, 0], dtype=np.uint8)
ivory = Material(Vec3(0.4, 0.4, 0.3) * np.uint8(255))
red_rubber = Material(Vec3(0.3, 0.1, 0.1) * np.uint8(255))
scene_objects = [Sphere(Vec3(1.0, 2.0, -5.0), 3, ivory), 
                 Sphere(Vec3(-3.0, -4.0, -10.0), 5, red_rubber)]
lights = [Light(Vec3(0.0, 0.0, 0.0), 1.5)]
def render(scene_objects, lights : List[Light], scene_background, img_w, img_h):
    img_matrix = np.ndarray((img_h,img_w, 3), dtype=np.uint8)
    fov = (110.0/360) * (2*math.pi)
    screen_dist = 1.0
    up_dir = (math.tan(fov/2) * screen_dist) / (img_h / 2.0)
    right_dir = (math.tan(fov/2) * screen_dist) / (img_w / 2.0)
    for i in range(img_h):
        for j in range(img_w):
            cam_direction = Vec3(right_dir * ((j + 0.5) - img_w/2),
                                 up_dir * (img_h/2 - (i + 0.5)), -1.0) 
            cam_direction.normalize()

            closest_object_idx = None
            closest_object_dist = np.inf
            closest_intersection_point = None
            for obj_idx in range(len(scene_objects)):
                obj = scene_objects[obj_idx]
                intersection_point = obj.intersect(Vec3(0.0,0.0,0.0), cam_direction)
                if intersection_point is not None:
                    if abs(intersection_point) < closest_object_dist:
                        closest_object_idx = obj_idx
                        closest_object_dist = abs(intersection_point)
                        closest_intersection_point = intersection_point
            if closest_object_idx is not None:
                for light in lights:
                    normal = scene_objects[closest_object_idx].normal_at(closest_intersection_point)
                    light_dir = light.position - closest_intersection_point
                    light_dir.normalize()
                    intensity = max(0, normal * light_dir) * light.intensity
                    color = scene_objects[closest_object_idx].material.diffuse_color * intensity
                    color = np.clip(color.to_numpy(), 0, 255).astype(np.uint8)
                    img_matrix[i][j] = color
            else:
                img_matrix[i][j] = scene_background

    return img_matrix

img_matrix = render(scene_objects, lights, scene_background, 1024, 768)
write_ppm(img_matrix, "img.ppm")