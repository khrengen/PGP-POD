#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <float.h>

#define _USE_MATH_DEFINES
#include <math.h>
#include <iostream>

#define CSC(call)                                                   \
do {                                                                \
    cudaError_t res = call;                                         \
    if (res != cudaSuccess) {                                       \
        fprintf(stderr, "ERROR in %s:%d. Message: %s\n",            \
                __FILE__, __LINE__, cudaGetErrorString(res));       \
        exit(0);                                                    \
    }                                                               \
} while(0)

const int trig_num = 355;// 2(floor), 8(oct) + 8*3*2(oct_edges), 20(icos) + 20*3*2(icos_edges), 12*3(dodec) + 12*5*2(dodec_edges)

using namespace std;

struct vec3 {
    float x;
    float y;
    float z;
};

struct vec4 {
    float diffus;
    float spec;
    float refl;
    float refr;
};

struct material {
    float refractive_index;
    float specular_exponent;
    uchar4 color;
    vec4 albedo;
};

struct trig {
    vec3 a;
    vec3 b;
    vec3 c;
    material mat;
    vec3 normal;
};

struct Light {
    vec3 pos;
    float intensity;
    vec3 clr;
};

__device__ __host__
float dot(const vec3 &a, const vec3 &b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ __host__
vec3 prod(const vec3 &a, const vec3 &b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

__device__ __host__
vec3 norm(const vec3 &v) {
    float l = sqrt(dot(v, v));
    return {v.x / l, v.y / l, v.z / l};
}

__device__ __host__
vec3 diff(const vec3 &a, const vec3 &b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

__device__ __host__
vec3 add(const vec3 &a, const vec3 &b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

__device__ __host__
vec3 mult(const vec3 &a, const vec3 &b, const vec3 &c, const vec3 &v) {
    return {    a.x * v.x + b.x * v.y + c.x * v.z,
                a.y * v.x + b.y * v.y + c.y * v.z,
                a.z * v.x + b.z * v.y + c.z * v.z };
}

__device__ __host__
vec3 mult_const(const vec3 &a, float c) {
    return {a.x * c, a.y * c, a.z * c};
}

__device__ __host__
vec3 get_norm(const vec3 &a, const vec3 &b, const vec3 &c) {
    return norm(prod(diff(b, a), diff(c, a)));
}

__device__ __host__
int Closest_trig(const vec3 &pos, const vec3 &dir, float &ts_min, trig* trigs) {
    int k_min = -1;
    for (int k = 0; k < trig_num; k++) {
        vec3 e1 = diff(trigs[k].b, trigs[k].a);
        vec3 e2 = diff(trigs[k].c, trigs[k].a);
        vec3 p = prod(dir, e2); 
        double div = dot(p, e1);
        if (fabs(div) < 1e-10) {
            continue;
        }
        vec3 t = diff(pos, trigs[k].a);
        double u = dot(p, t) / div;
        if (u < 0.0 || u > 1.0) {
            continue;
        }
        vec3 q = prod(t, e1);
        double v = dot(q, dir) / div;
        if (v < 0.0 || v + u > 1.0) {
            continue;
        }
        double ts = dot(q, e2) / div;
        if (ts < 0.0) {
            continue;
        }
        if (k_min == -1 || ts < ts_min) {
            k_min = k;
            ts_min = ts;
        }
    }
    return k_min;
}

__device__ __host__
vec3 reflect(const vec3 &I, const vec3 &N) {
    return diff(I, mult_const(mult_const(N, 2), dot(I,N)));
}

__device__ __host__
vec3 refract(const vec3 &I, const vec3 &N, const float eta_t, const float eta_i=1.0) {
    float cosi = -max(-1.0, min(1.0, dot(I,N)));
    if (cosi < 0) {
        return refract(I, mult_const(N, -1), eta_i, eta_t);
    } 
    float eta = eta_i / eta_t;
    float k = abs(1 - eta*eta*(1 - cosi*cosi));
    return k < 0 ? vec3{10,0,0} : norm(add(mult_const(I,eta), mult_const(N,(eta*cosi - sqrt(k)))));
}

__device__ __host__
uchar4 get_texture_clr(uchar4 *floor, float x, float y, int w, int floor_w) {
    x = (x / floor_w*w + w/2);
    y = (y / floor_w*w + w/2);
    return floor[(int)x * w + (int)y];
}

void floor_painting(uchar4 *floor, vec3 color, int w, int h) {
    for (int i = 0; i < w * h; i++) {
        floor[i] = make_uchar4(color.x * floor[i].x, color.y * floor[i].y, color.z * floor[i].z, floor[i].w);
    }
}

void build_edges(int first_trig, int trig_num, int &cur_poly, trig* trigs) {
    const material edge = {1., 125.0, {70, 70, 70, 0}, {0.6,  0.6, 0., 0.}};  //edge
    for (int i = first_trig; i < first_trig + trig_num; i++) {
        vec3 N = get_norm(trigs[i].a, trigs[i].b, trigs[i].c); //нормаль

        vec3 d1 = add(trigs[i].a, mult_const(diff(trigs[i].b, trigs[i].a), 0.05));
        vec3 d2 = add(trigs[i].c, mult_const(diff(trigs[i].b, trigs[i].c), 0.05));
        vec3 d3 = add(trigs[i].a, mult_const(diff(trigs[i].c, trigs[i].a), 0.05));
        vec3 d4 = add(trigs[i].b, mult_const(diff(trigs[i].c, trigs[i].b), 0.05));
        vec3 d5 = add(trigs[i].c, mult_const(diff(trigs[i].a, trigs[i].c), 0.05));
        vec3 d6 = add(trigs[i].b, mult_const(diff(trigs[i].a, trigs[i].b), 0.05));

        trigs[cur_poly] =     {add(trigs[i].a, mult_const(N,1e-4)), add(d1, mult_const(N,1e-4)), add(d2, mult_const(N,1e-4)), edge};
        trigs[cur_poly + 1] = {add(trigs[i].a, mult_const(N,1e-4)), add(d2, mult_const(N,1e-4)), add(trigs[i].c, mult_const(N,1e-4)), edge};
        trigs[cur_poly + 2] = {add(trigs[i].b, mult_const(N,1e-5)), add(d4, mult_const(N,1e-5)), add(d3, mult_const(N,1e-5)), edge};
        trigs[cur_poly + 3] = {add(trigs[i].b, mult_const(N,1e-5)), add(d3, mult_const(N,1e-5)), add(trigs[i].a, mult_const(N,1e-5)), edge};
        trigs[cur_poly + 4] = {add(trigs[i].c, mult_const(N,1e-3)), add(d5, mult_const(N,1e-3)), add(d6, mult_const(N,1e-3)), edge};
        trigs[cur_poly + 5] = {add(trigs[i].c, mult_const(N,1e-3)), add(d6, mult_const(N,1e-3)), add(trigs[i].b, mult_const(N,1e-3)), edge};

        trigs[i].normal = N;
        cur_poly+= 6;
    }
}

void build_dodec_edges(int first_trig, int trig_num, int &cur_poly, trig* trigs) {
    const material edge = {1., 125.0, {70, 70, 70, 0}, {0.6,  0.6, 0., 0.}};  //edge
    int count = 0;
    for (int i = first_trig; i < first_trig + trig_num; i++, count++) {
        vec3 N = get_norm(trigs[i].a, trigs[i].b, trigs[i].c);
        if (count % 3 == 0) {
            vec3 d1 = add(trigs[i].a, mult_const(diff(trigs[i].b, trigs[i].a), 0.05));
            vec3 d2 = add(trigs[i].c, mult_const(diff(trigs[i].b, trigs[i].c), 0.05));
            trigs[cur_poly] =     {add(trigs[i].a, mult_const(N,1e-4)), add(d1, mult_const(N,1e-4)), add(d2, mult_const(N,1e-4)), edge};
            trigs[cur_poly + 1] = {add(trigs[i].a, mult_const(N,1e-4)), add(d2, mult_const(N,1e-4)), add(trigs[i].c, mult_const(N,1e-4)), edge};
            cur_poly+= 2;
        } else {
            vec3 d3 = add(trigs[i].a, mult_const(diff(trigs[i].c, trigs[i].a), 0.05));
            vec3 d4 = add(trigs[i].b, mult_const(diff(trigs[i].c, trigs[i].b), 0.05));
            vec3 d5 = add(trigs[i].c, mult_const(diff(trigs[i].a, trigs[i].c), 0.05));
            vec3 d6 = add(trigs[i].b, mult_const(diff(trigs[i].a, trigs[i].b), 0.05));
            trigs[cur_poly] = {add(trigs[i].b, mult_const(N,1e-5)), add(d4, mult_const(N,1e-5)), add(d3, mult_const(N,1e-5)), edge};
            trigs[cur_poly + 1] = {add(trigs[i].b, mult_const(N,1e-5)), add(d3, mult_const(N,1e-5)), add(trigs[i].a, mult_const(N,1e-5)), edge};
            trigs[cur_poly + 2] = {add(trigs[i].c, mult_const(N,1e-3)), add(d5, mult_const(N,1e-3)), add(d6, mult_const(N,1e-3)), edge};
            trigs[cur_poly + 3] = {add(trigs[i].c, mult_const(N,1e-3)), add(d6, mult_const(N,1e-3)), add(trigs[i].b, mult_const(N,1e-3)), edge};
            cur_poly+= 4;
        }
        trigs[i].normal = N;
    }
}

void Octahedron(float radius, float c_x, float c_y, float c_z, const material &mat, trig* trigs) {
    vec3 coords[6];
    coords[0] = {c_x, c_y - radius, c_z};
    coords[1] = {c_x - radius, c_y, c_z};
    coords[2] = {c_x, c_y, c_z - radius};
    coords[3] = {c_x + radius, c_y, c_z};
    coords[4] = {c_x, c_y, c_z + radius};
    coords[5] = {c_x, c_y + radius, c_z};

    trigs[2] = {coords[0], coords[1], coords[2], mat};
    trigs[3] = {coords[0], coords[2], coords[3], mat};
    trigs[4] = {coords[0], coords[3], coords[4], mat};
    trigs[5] = {coords[0], coords[4], coords[1], mat};
    trigs[6] = {coords[5], coords[2], coords[1], mat};
    trigs[7] = {coords[5], coords[3], coords[2], mat};
    trigs[8] = {coords[5], coords[4], coords[3], mat};
    trigs[9] = {coords[5], coords[1], coords[4], mat};
}

void Icosahedron(float radius, float c_x, float c_y, float c_z, const material &mat, trig* trigs) {
    float magicAngle = M_PI * 26.565 / 180;
    float segmentAngle = M_PI * 72 / 180;
    float currentAngle = 0.0;

    vec3 coords[12];
    coords[0] =  {c_x, c_y + radius, c_z};
    coords[11] = {c_x, c_y - radius, c_z};
            
    for (int  i = 1; i < 6; i++) {
        coords[i] = {c_x + radius * sin(currentAngle) * cos(magicAngle),
                     c_y + radius * sin(magicAngle),
                     c_z + radius * cos(currentAngle) * cos(magicAngle)};
        currentAngle += segmentAngle;
    }
    currentAngle = M_PI * 36 / 180;
    for (int i = 6; i < 11; i++) {
        coords[i] = {c_x + radius * sin(currentAngle) * cos(-magicAngle),
                     c_y + radius * sin(-magicAngle),
                     c_z + radius * cos(currentAngle) * cos(-magicAngle)};
        currentAngle += segmentAngle;
    }

    trigs[10] = {coords[0], coords[1], coords[2], mat};
    trigs[11] = {coords[0], coords[2], coords[3], mat};
    trigs[12] = {coords[0], coords[3], coords[4], mat};
    trigs[13] = {coords[0], coords[4], coords[5], mat};
    trigs[14] = {coords[0], coords[5], coords[1], mat};

    trigs[15] = {coords[11], coords[7], coords[6], mat};
    trigs[16] = {coords[11], coords[8], coords[7], mat};
    trigs[17] = {coords[11], coords[9], coords[8], mat};
    trigs[18] = {coords[11], coords[10], coords[9], mat};
    trigs[19] = {coords[11], coords[6], coords[10], mat};

    trigs[20] = {coords[2], coords[1], coords[6], mat};
    trigs[21] = {coords[3], coords[2], coords[7], mat};
    trigs[22] = {coords[4], coords[3], coords[8], mat};
    trigs[23] = {coords[5], coords[4], coords[9], mat};
    trigs[24] = {coords[1], coords[5], coords[10], mat};

    trigs[25] = {coords[6], coords[7], coords[2], mat};
    trigs[26] = {coords[7], coords[8], coords[3], mat};
    trigs[27] = {coords[8], coords[9], coords[4], mat};
    trigs[28] = {coords[9], coords[10], coords[5], mat};
    trigs[29] = {coords[10], coords[6], coords[1], mat};
}

void Dodecahedron(float radius, float c_x, float c_y, float c_z, const material &mat, trig* trigs) {
    float phi = (1.0 + sqrt(5.0)) / 2.0;
    vec3 coords[20] = {{c_x + (-1.f / phi / sqrt(3.f) * radius), c_y, c_z + (phi / sqrt(3.f) * radius)},
                       {c_x + (1.f / phi / sqrt(3.f) * radius), c_y, c_z + (phi / sqrt(3.f) * radius)},
                       {c_x + (-1.f / sqrt(3.f) * radius), c_y + (1.f / sqrt(3.f) * radius), c_z + (1.f / sqrt(3.f) * radius)},
                       {c_x + (1.f / sqrt(3.f) * radius), c_y + (1.f / sqrt(3.f) * radius), c_z + (1.f / sqrt(3.f) * radius)},
                       {c_x + (1.f / sqrt(3.f) * radius), c_y + (-1.f / sqrt(3.f) * radius), c_z + (1.f / sqrt(3.f) * radius)},
                       {c_x + (-1.f / sqrt(3.f) * radius), c_y + (-1.f / sqrt(3.f) * radius), c_z + (1.f / sqrt(3.f) * radius)},
                       {c_x, c_y + (-phi / sqrt(3.f) * radius), c_z + (1.f / phi / sqrt(3.f) * radius)},
                       {c_x, c_y + (phi / sqrt(3.f) * radius), c_z + (1.f / phi / sqrt(3.f) * radius)},
                       {c_x + (-phi / sqrt(3.f) * radius), c_y + (-1.f / phi / sqrt(3.f) * radius), c_z},
                       {c_x + (-phi / sqrt(3.f) * radius), c_y + (1.f / phi / sqrt(3.f) * radius), c_z},
                       {c_x + (phi / sqrt(3.f) * radius), c_y + (1.f / phi / sqrt(3.f) * radius), c_z},
                       {c_x + (phi / sqrt(3.f) * radius), c_y + (-1.f / phi / sqrt(3.f) * radius), c_z},
                       {c_x, c_y + (-phi / sqrt(3.f) * radius), c_z + (-1.f / phi / sqrt(3.f) * radius)},
                       {c_x, c_y + (phi / sqrt(3.f) * radius), c_z + (-1.f / phi / sqrt(3.f) * radius)},
                       {c_x + (1.f / sqrt(3.f) * radius), c_y + (1.f / sqrt(3.f) * radius), c_z + (-1.f / sqrt(3.f) * radius)},
                       {c_x + (1.f / sqrt(3.f) * radius), c_y + (-1.f / sqrt(3.f) * radius), c_z + (-1.f / sqrt(3.f) * radius)},
                       {c_x + (-1.f / sqrt(3.f) * radius), c_y + (-1.f / sqrt(3.f) * radius), c_z + (-1.f / sqrt(3.f) * radius)},
                       {c_x + (-1.f / sqrt(3.f) * radius), c_y + (1.f / sqrt(3.f) * radius), c_z + (-1.f / sqrt(3.f) * radius)},
                       {c_x + (1.f / phi / sqrt(3.f) * radius), c_y, c_z + (-phi / sqrt(3.f) * radius)},
                       {c_x + (-1.f / phi / sqrt(3.f) * radius), c_y, c_z + (-phi / sqrt(3.f) * radius)}};

    trigs[30] = {coords[4],  coords[0],  coords[6],  mat};
    trigs[31] = {coords[0],  coords[5],  coords[6],  mat};
    trigs[32] = {coords[4],  coords[1],  coords[0],  mat};

    trigs[33] = {coords[7],  coords[0],  coords[3],  mat};
    trigs[34] = {coords[0],  coords[1],  coords[3],  mat};
    trigs[35] = {coords[7],  coords[2],  coords[0],  mat};

    trigs[36] = {coords[10], coords[1],  coords[11], mat};
    trigs[37] = {coords[1],  coords[4],  coords[11], mat};
    trigs[38] = {coords[10], coords[3],  coords[1],  mat};

    trigs[39] = {coords[8],  coords[0],  coords[9],  mat};
    trigs[40] = {coords[0],  coords[2],  coords[9],  mat};
    trigs[41] = {coords[8],  coords[5],  coords[0],  mat};

    trigs[42] = {coords[12], coords[5],  coords[16], mat};
    trigs[44] = {coords[5],  coords[8],  coords[16], mat};
    trigs[43] = {coords[12], coords[6],  coords[5],  mat};

    trigs[45] = {coords[15], coords[4],  coords[12], mat};
    trigs[46] = {coords[4],  coords[6],  coords[12], mat};
    trigs[47] = {coords[15], coords[11], coords[4],  mat};

    trigs[48] = {coords[17], coords[2],  coords[13], mat};
    trigs[49] = {coords[2],  coords[7],  coords[13], mat};
    trigs[50] = {coords[17], coords[9],  coords[2],  mat};

    trigs[51] = {coords[13], coords[3],  coords[14], mat};
    trigs[52] = {coords[3],  coords[10], coords[14], mat};
    trigs[53] = {coords[13], coords[7],  coords[3],  mat};

    trigs[54] = {coords[19], coords[8],  coords[17], mat};
    trigs[55] = {coords[8],  coords[9],  coords[17], mat};
    trigs[56] = {coords[19], coords[16], coords[8],  mat};

    trigs[57] = {coords[14], coords[11], coords[18], mat};
    trigs[58] = {coords[11], coords[15], coords[18], mat};
    trigs[59] = {coords[14], coords[10], coords[11], mat};

    trigs[60] = {coords[18], coords[12], coords[19], mat};
    trigs[61] = {coords[12], coords[16], coords[19], mat};
    trigs[62] = {coords[18], coords[15], coords[12], mat};
    
    trigs[63] = {coords[19], coords[13], coords[18], mat};
    trigs[64] = {coords[13], coords[14], coords[18], mat};
    trigs[65] = {coords[19], coords[17], coords[13], mat};
}

void build_space(trig* trigs, const vec3 &oct_c, const vec3 &ico_c, const vec3 &dodec_c, const trig &fl1, const trig &fl2, const vec3 &rads, material *materials) {
    int cur_free_poly = 66; //for edges after figures

    trigs[0] = {fl1.a, fl1.b, fl1.c, materials[3], {0.0, 0.0, 1.0}};
    trigs[1] = {fl2.a, fl2.b, fl2.c, materials[3], {0.0, 0.0, 1.0}};

    Octahedron(rads.x, oct_c.x, oct_c.y, oct_c.z, materials[0], trigs);
    Icosahedron(rads.y, ico_c.x, ico_c.y, ico_c.z, materials[1], trigs);
    Dodecahedron(rads.z, dodec_c.x, dodec_c.y, dodec_c.z, materials[2], trigs);

    build_edges(2, 28, cur_free_poly, trigs);
    build_dodec_edges(30, 36, cur_free_poly, trigs);
}

__device__ __host__
uchar4 ray(vec3 pos, vec3 dir, int depth, uchar4 *floor, int f_size, int f_txt_size, trig* trigs, Light *lights, int l_num, int max_deep) {
    uchar4 reflect_color = {0, 0, 0, 0};
    uchar4 refract_color = {0, 0, 0, 0};

    if (depth > max_deep) {
        return {0, 0, 0, 0};
    }
    //пересечение с полигоном
    float ts_min = FLT_MAX;
    int k_min = Closest_trig(pos, dir, ts_min, trigs);
    if (k_min == -1) {
        return {0, 0, 0, 0};
    }

    //точка пересечение луча и ближ полигона
    vec3 z = add(pos, mult_const(dir, ts_min));

    material cur_mat = trigs[k_min].mat;
    uchar4 clr = make_uchar4(cur_mat.color.x, cur_mat.color.y, cur_mat.color.z, cur_mat.color.w);

    //наложение текстуры на пол
    if (k_min < 2) {
        clr = get_texture_clr(floor, z.x, z.y, f_txt_size, f_size);
    }

    // нормаль ближайшего
    vec3 N = trigs[k_min].normal;

    //отражение преломление 
    vec3 reflect_dir = norm(reflect(dir, N));
    vec3 refract_dir = refract(dir, N, cur_mat.refractive_index);
    vec3 z_near_rfl = dot(reflect_dir, N) < 0 ? diff(z, mult_const(N, 1e-6)) : add(z, mult_const(N, 1e-6));
    reflect_color = ray(z_near_rfl, reflect_dir, depth + 1, floor, f_size, f_txt_size, trigs, lights, l_num, max_deep);

    if (refract_dir.x <= 1.0) { // если нет полного отражения
        vec3 z_near_rfr = dot(norm(refract_dir), N) < 0 ? diff(z, mult_const(N, 1e-6)) : add(z, mult_const(N, 1e-6));
        refract_color = ray(z_near_rfr, norm(refract_dir), depth + 1, floor, f_size, f_txt_size, trigs, lights, l_num, max_deep);
    }
    
    vec3 sum_clr = {0.0, 0.0, 0.0};
    float ambient_intens = 0.1;

    for (int i = 0; i < l_num; i++) {
        float diffuse_intens = 0.0;
        float specular_intens = 0.0;
        //тень
        bool shadow = false;
        vec3 light_dir = norm(diff(lights[i].pos, z));
        vec3 to_light_dir = mult_const(light_dir, -1);

        ts_min = FLT_MAX;
        int l_min = Closest_trig(lights[i].pos, to_light_dir, ts_min, trigs);
        if (l_min != k_min) {
            //точка в тени
            shadow = true;
        }

        if (!shadow) {
            diffuse_intens = lights[i].intensity * max(0.0, dot(light_dir, N));
            specular_intens = pow(max(0.0, dot(mult_const(reflect(mult_const(light_dir, -1), N), -1), dir)), (double)cur_mat.specular_exponent) * lights[i].intensity;
        }
        sum_clr.x += lights[i].clr.x * clr.x*diffuse_intens*cur_mat.albedo.diffus + 255.0 * specular_intens*cur_mat.albedo.spec;
        sum_clr.y += lights[i].clr.y * clr.y*diffuse_intens*cur_mat.albedo.diffus + 255.0 * specular_intens*cur_mat.albedo.spec;
        sum_clr.z += lights[i].clr.z * clr.z*diffuse_intens*cur_mat.albedo.diffus + 255.0 * specular_intens*cur_mat.albedo.spec;
    }

    clr = make_uchar4(min(ambient_intens*clr.x + sum_clr.x + cur_mat.albedo.refl*reflect_color.x + cur_mat.albedo.refr*refract_color.x, 255.0),
                      min(ambient_intens*clr.y + sum_clr.y + cur_mat.albedo.refl*reflect_color.y + cur_mat.albedo.refr*refract_color.y, 255.0),
                      min(ambient_intens*clr.z + sum_clr.z + cur_mat.albedo.refl*reflect_color.z + cur_mat.albedo.refr*refract_color.z, 255.0),
                      clr.w);
    return clr;

}

void cpu_ssaa(uchar4* data, uchar4* out, int w, int h, int sqrt_ray_num) {

    int w_scale = w * sqrt_ray_num;
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            int y_scale = i * sqrt_ray_num;
            int x_scale = j * sqrt_ray_num;
            uint4 aver_clr = make_uint4(0, 0, 0, 0);

            for (int n = y_scale; n < y_scale + sqrt_ray_num; n++) {
                for (int m = x_scale; m < x_scale + sqrt_ray_num; m++) {
                    aver_clr.x += data[n * w_scale + m].x;
                    aver_clr.y += data[n * w_scale + m].y;
                    aver_clr.z += data[n * w_scale + m].z;
                }
            }
            float d = sqrt_ray_num * sqrt_ray_num;
            out[i*w + j] = make_uchar4(aver_clr.x / d, aver_clr.y / d, aver_clr.z / d, aver_clr.w);
        }
    }
}


__global__ void kernel_ssaa(uchar4* data, uchar4* out, int w, int h, int sqrt_ray_num) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y;

    int w_scale = w * sqrt_ray_num;
    for (int i = idx; i < h; i+= offsetx) {
        for (int j = idy; j < w; j+= offsety) {
            int y_scale = i * sqrt_ray_num;
            int x_scale = j * sqrt_ray_num;
            uint4 aver_clr = make_uint4(0, 0, 0, 0);

            for (int n = y_scale; n < y_scale + sqrt_ray_num; n++) {
                for (int m = x_scale; m < x_scale + sqrt_ray_num; m++) {
                    aver_clr.x += data[n * w_scale + m].x;
                    aver_clr.y += data[n * w_scale + m].y;
                    aver_clr.z += data[n * w_scale + m].z;
                }
            }
            float d = sqrt_ray_num * sqrt_ray_num;
            out[i*w + j] = make_uchar4(aver_clr.x / d, aver_clr.y / d, aver_clr.z / d, aver_clr.w);
        }
    }
}

void render(vec3 pc, vec3 pv, int w, int h, double angle, uchar4 *data, uchar4 *floor, int f_size, int f_txt_size, trig* trigs, Light *lights, int lights_num, int max_deep) {
    float dw = 2.0 / (w - 1);
    float dh = 2.0 / (h - 1);
    float z = 1.0 / tan(angle * M_PI / 360.0);
    vec3 bz = norm(diff(pv, pc));
    vec3 bx = norm(prod(bz, {0.0, 0.0, 1.0}));
    vec3 by = prod(bx, bz);
    for (int i = 0; i < w; i++) {
        for (int j = 0; j < h; j++) {
            vec3 v = {-1.f + dw * i, (-1.f + dh * j) * h / w, z};
            vec3 dir = norm(mult(bx, by, bz, v));
            data[(h - 1 - j) * w + i] = ray(pc, dir, 0, floor, f_size, f_txt_size, trigs, lights, lights_num, max_deep);
        }
    }
}

__global__ void kernel_render(vec3 pc, vec3 pv, int w, int h, double angle, uchar4 *data, uchar4 *floor, int f_size, int f_txt_size, trig* trigs, Light *lights, int l_num, int max_deep) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y;

    float dw = 2.0 / (w - 1);
    float dh = 2.0 / (h - 1);
    float z = 1.0 / tan(angle * M_PI / 360.0);
    vec3 bz = norm(diff(pv, pc));
    vec3 bx = norm(prod(bz, {0.0, 0.0, 1.0}));
    vec3 by = prod(bx, bz);
    for (int j = idy; j < h; j+= offsety) {
        for (int i = idx; i < w; i+= offsetx) {
            vec3 v = {-1.f + dw * i, (-1.f + dh * j) * h / w, z};
            vec3 dir = norm(mult(bx, by, bz, v));
            data[(h - 1 - j) * w + i] = ray(pc, dir, 0, floor, f_size, f_txt_size, trigs, lights, l_num, max_deep);
        }
    }
}

int main(int argc, char *argv[]) {

    bool on_gpu = true;

    int frame_num = 120;
    char path_to_image[100] = "res/%d.data";
    int w = 640, h = 480, angle = 120;
    float rc0 = 7.0, zc0 = 3.0, phic0 = 0.0, acr = 2.0, acz = 1.0, wcr = 2.0, wcz = 6.0, wcphi = 1.0, pcr = 0.0, pcz = 0.0;
    float rn0 = 2.0, zn0 = 0.0, phin0 = 0.0, anr = 0.5, anz = 0.1, wnr = 1.0, wnz = 4.0, wnphi = 1.0, pnr = 0.0, pnz = 0.0;

    float oct_cnt_x = 3.0, oct_cnt_y = -2.0, oct_cnt_z = 2.5,  oct_col_r = 1.0, oct_col_g = 0.0, oct_col_b = 1.0, oct_r = 1.5, oct_refl = 0.2, oct_tran = 0.8;
    float dod_cnt_x = -1.0, dod_cnt_y = -1.5, dod_cnt_z = 1.5, dod_col_r = 0.2, dod_col_g = 1.0, dod_col_b = 0.2, dod_r = 1.5, dod_refl = 0.5, dod_tran = 0.0;
    float ico_cnt_x = -2.5, ico_cnt_y = 2.5, ico_cnt_z = 2.5,  ico_col_r = 0.0, ico_col_g = 1.0, ico_col_b = 1.0, ico_r = 1.5, ico_refl = 0.2, ico_tran = 0.9;
    int oct_l_num = 0, dod_l_num = 0, ico_l_num = 0;

    float f1_x = -5.0, f1_y = -5.0, f1_z = 0.0, f2_x = -5.0, f2_y = 5.0, f2_z = 0.0, f3_x = 5.0, f3_y = 5.0, f3_z= 0.0, f4_x = 5.0, f4_y = -5.0, f4_z = 0.0;
    char path_to_floor[100] = "board.data";
    float f_col_r = 1.0, f_col_g = 1.0, f_col_b = 1.0, f_refl = 0.7;

    int lights_num = 1;
    float l_pos_x = 5.0, l_pos_y = 5.0, l_pos_z = 5.0;
    float light_r = 1.0, light_g = 1.0, light_b = 1.0, light_int = 1.6;

    int max_deep = 2, sqrt_ray_num = 4;

    if (argc > 2) {
        cerr << "Wrong input: too many args\n";
        exit(-1);
    } else if (argc == 2) {
        string arg = argv[1];
        if (arg == "--deafult") {
            cout << frame_num << '\n';
            cout << path_to_image << '\n';
            cout << w << ' ' << h << ' ' << angle << '\n';
            cout << rc0 << ' ' <<  zc0  << ' ' << phic0 << ' ' << acr << ' ' << acz << ' ' << wcr << ' ' << wcz <<' ' << wcphi << ' ' << pcr << ' ' << pcz << '\n';
            cout << rn0 << ' ' << zn0 << ' ' << phin0 << ' ' << anr << ' ' << anz << ' ' << wnr << ' ' << wnz << ' ' << wnphi << ' ' << pnr << ' ' << pnz << '\n';
            cout << oct_cnt_x << ' ' << oct_cnt_y << ' ' << oct_cnt_z << ' ' << oct_col_r << ' ' << oct_col_g << ' ' << oct_col_b << ' ' << oct_r << ' ' << oct_refl << ' ' << oct_tran << ' ' << oct_l_num << '\n';
            cout << dod_cnt_x << ' ' << dod_cnt_y << ' ' << dod_cnt_z << ' ' << dod_col_r << ' ' << dod_col_g << ' ' << dod_col_b << ' ' << dod_r << ' ' << dod_refl << ' ' << dod_tran << ' ' << dod_l_num << '\n';
            cout << ico_cnt_x << ' ' << ico_cnt_y << ' ' << ico_cnt_z << ' ' << ico_col_r << ' ' << ico_col_g << ' ' << ico_col_b << ' ' << ico_r << ' ' << ico_refl << ' ' << ico_tran << ' ' << ico_l_num << '\n';
            cout << f1_x << ' ' << f1_y << ' ' << f1_z << ' ' << f2_x << ' ' << f2_y << ' ' << f2_z << ' ' << f3_x << ' ' << f3_y << ' ' << f3_z << ' ' << f4_x << ' ' << f4_y << ' ' << f4_z << '\n';
            cout << path_to_floor << '\n';
            cout << f_col_r << ' ' << f_col_g << ' ' << f_col_b << ' ' << f_refl << '\n';
            cout << lights_num << '\n';
            cout << l_pos_x << ' ' << l_pos_y << ' ' << l_pos_z << '\n';
            cout << light_r << ' ' << light_g << ' ' << light_b << '\n';
            cout << max_deep << ' ' << sqrt_ray_num << '\n';
            return 0;

        } else if (arg == "--cpu") {
            on_gpu = false;
        }
    }

    cin >> frame_num;
    cin >> path_to_image;
    cin >> w >> h >> angle;
    cin >> rc0 >> zc0 >> phic0 >> acr >> acz >> wcr >> wcz >> wcphi >> pcr >> pcz;
    cin >> rn0 >> zn0 >> phin0 >> anr >> anz >> wnr >> wnz >>wnphi >> pnr >> pnz;
    cin >> oct_cnt_x >> oct_cnt_y >> oct_cnt_z >> oct_col_r >> oct_col_g >> oct_col_b >> oct_r >> oct_refl >> oct_tran >> oct_l_num;
    cin >> dod_cnt_x >> dod_cnt_y >> dod_cnt_z >> dod_col_r >> dod_col_g >> dod_col_b >> dod_r >> dod_refl >> dod_tran >> dod_l_num;
    cin >> ico_cnt_x >> ico_cnt_y >> ico_cnt_z >> ico_col_r >> ico_col_g >> ico_col_b >> ico_r >> ico_refl >> ico_tran >> ico_l_num;
    cin >> f1_x >> f1_y >> f1_z >> f2_x >> f2_y >> f2_z >> f3_x >> f3_y >> f3_z >> f4_x >> f4_y >> f4_z;
    cin >> path_to_floor;
    cin >> f_col_r >> f_col_g >> f_col_b >> f_refl;
    cin >> lights_num;

    Light *lights = (Light*)malloc(sizeof(Light) * lights_num);
    for (int i = 0; i < lights_num; i++) {
        cin >> l_pos_x >> l_pos_y >> l_pos_z;
        cin >> light_r >> light_g >> light_b;
        lights[i] = {{l_pos_x, l_pos_y, l_pos_z}, light_int, {light_r, light_g, light_b}};
    }
    cin >> max_deep >> sqrt_ray_num;
    
    char buff[256];
    uchar4 *data = (uchar4*)malloc(sizeof(uchar4) * w * h * sqrt_ray_num * sqrt_ray_num);
    uchar4 *short_data = (uchar4*)malloc(sizeof(uchar4) * w * h);
    trig *trigs = (trig*)malloc(sizeof(trig) * trig_num);

    uchar4 *dev_data;
    uchar4 *dev_short_data;
    trig *dev_trigs;
    Light *dev_lights;
    uchar4 *dev_floor;

    vec3 pc, pv;

    int w1, h1;
    FILE *fp = fopen(path_to_floor, "rb");
    fread(&w1, sizeof(int), 1, fp);
    fread(&h1, sizeof(int), 1, fp);
    uchar4 *floor = (uchar4 *)malloc(sizeof(uchar4) * w1 * h1);
    fread(floor, sizeof(uchar4), w1 * h1, fp);
    fclose(fp);
    floor_painting(floor, {f_col_r, f_col_g, f_col_b}, w1, h1);
    int f_size = (int)abs(f1_x - f3_x);

    trig fl1 = {{f1_x, f1_y, f1_z}, {f2_x, f2_y, f2_z}, {f3_x, f3_y, f3_z}};
    trig fl2 = {{f1_x, f1_y, f1_z}, {f3_x, f3_y, f3_z}, {f4_x, f4_y, f4_z}};
    material *materials = (material*)malloc(sizeof(material) * 4);
    materials[0] = {1., 125.0, make_uchar4(oct_col_r * 255, oct_col_g * 255, oct_col_b * 255, 0), {0.2,  0.4, oct_refl, oct_tran}}; // purple_oct
    materials[1] = {1., 125.0, make_uchar4(ico_col_r * 255, ico_col_g * 255, ico_col_b * 255, 0), {0.2,  0.3, ico_refl, ico_tran}}; // blue_ico
    materials[2] = {1., 125.0, make_uchar4(dod_col_r * 255, dod_col_g * 255, dod_col_b * 255, 0), {0.7,  0.5, dod_refl, dod_tran}}; //green_dod
    materials[3] = {1., 125.0, make_uchar4(f_col_r * 255, f_col_g * 255, f_col_b * 255, 0), {0.6,  0.7, f_refl, 0.0}};              //blik_floor
    build_space(trigs, {oct_cnt_x, oct_cnt_y, oct_cnt_z}, {ico_cnt_x, ico_cnt_y, ico_cnt_z}, {dod_cnt_x, dod_cnt_y, dod_cnt_z}, fl1, fl2, {oct_r, ico_r, dod_r}, materials);

    if (on_gpu) {
        CSC(cudaMalloc(&dev_data, sizeof(uchar4) * w * h * sqrt_ray_num * sqrt_ray_num));
        CSC(cudaMalloc(&dev_short_data, sizeof(uchar4) * w * h));
        CSC(cudaMalloc(&dev_trigs, sizeof(trig) * trig_num));
        CSC(cudaMalloc(&dev_lights, sizeof(Light) * lights_num));
        CSC(cudaMalloc(&dev_floor, sizeof(uchar4) * w1 * h1));

        CSC(cudaMemcpy(dev_trigs, trigs, sizeof(trig) * trig_num, cudaMemcpyHostToDevice));
        CSC(cudaMemcpy(dev_lights, lights, sizeof(Light) * lights_num, cudaMemcpyHostToDevice));
        CSC(cudaMemcpy(dev_floor, floor, sizeof(uchar4) * w1 * h1, cudaMemcpyHostToDevice));
    }

    for (int k = 0; k < frame_num; k++) {
        float step = 2 * M_PI * k / frame_num; 

        float rct = rc0 + acr * sin(wcr * step + pcr);
        float zct = zc0 + acz * sin(wcz * step + pcz);
        float phict = phic0 + wcphi * step;

        float rnt = rn0 + anr * sin(wnr * step + pnr);
        float znt = zn0 + anz * sin(wnz * step + pnz);
        float phint = phin0 + wnphi * step;

        pc = {rct * cos(phict), rct * sin(phict), zct};
        pv = {rnt * cos(phint), rnt * sin(phint), znt};

        cudaEvent_t begin, end;
        float gpu_time = 0.0;
        cudaEventCreate(&begin);
        cudaEventCreate(&end);
        cudaEventRecord(begin, 0);

        if (on_gpu) {
            kernel_render<<<dim3(8, 32), dim3(8, 32)>>>(pc, pv, w * sqrt_ray_num, h * sqrt_ray_num, angle, dev_data, dev_floor, f_size, w1, dev_trigs, dev_lights, lights_num, max_deep);
            CSC(cudaGetLastError());

            kernel_ssaa<<<dim3(8, 32), dim3(8, 32)>>>(dev_data, dev_short_data, w, h, sqrt_ray_num);
            CSC(cudaGetLastError());
            CSC(cudaMemcpy(data, dev_short_data, sizeof(uchar4) * w * h, cudaMemcpyDeviceToHost));

        } else {
            render(pc, pv, w * sqrt_ray_num, h * sqrt_ray_num, angle, data, floor, f_size, w1, trigs, lights, lights_num, max_deep);
            cpu_ssaa(data, short_data, w, h, sqrt_ray_num);
            memcpy(data, short_data, sizeof(uchar4) * w * h);
        }

        cudaEventRecord(end, 0);
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&gpu_time, begin, end);

        cout << k << '\t' << gpu_time << '\t' << w*h*sqrt_ray_num*sqrt_ray_num << '\n';
        sprintf(buff, path_to_image, k);
        //printf("%d: %s\n", k, buff);      
        FILE *out = fopen(buff, "wb");
        fwrite(&w, sizeof(int), 1, out);
        fwrite(&h, sizeof(int), 1, out);    
        fwrite(data, sizeof(uchar4), w * h, out);
        fclose(out);
    }
    free(data);
    free(short_data);
    free(floor);
    free(trigs);
    free(materials);

    if (on_gpu) {
        CSC(cudaFree(dev_data));
        CSC(cudaFree(dev_short_data));
        CSC(cudaFree(dev_trigs));
        CSC(cudaFree(dev_lights));
        CSC(cudaFree(dev_floor));
    }
    return 0;
}