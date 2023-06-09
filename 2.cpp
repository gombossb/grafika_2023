//=============================================================================================
// Mintaprogram: Zöld háromszög. Ervenyes 2019. osztol.
// + Computer Graphics Sample Program: Ray-tracing-let
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Gombos Sa'ndor Bence
// Neptun : ******
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
#include "framework.h"

// source: https://people.sc.fsu.edu/~jburkardt/data/obj/obj.html
// open cube without normals
std::string openCubeObj = R"(v 0.0 0.0 0.0
v 0.0 0.0 1.0
v 0.0 1.0 0.0
v 0.0 1.0 1.0
v 1.0 0.0 0.0
v 1.0 0.0 1.0
v 1.0 1.0 0.0
v 1.0 1.0 1.0
f 1 7 5
f 1 3 7
f 1 4 3
f 1 2 4
f 3 8 7
f 3 4 8
#f 5 7 8
#f 5 8 6
f 1 5 6
f 1 6 2
#f 2 6 8
#f 2 8 4
)";

std::string dodecahedronObj = R"(v -0.57735 -0.57735 0.57735
v 0.934172 0.356822 0
v 0.934172 -0.356822 0
v -0.934172 0.356822 0
v -0.934172 -0.356822 0
v 0 0.934172 0.356822
v 0 0.934172 -0.356822
v 0.356822 0 -0.934172
v -0.356822 0 -0.934172
v 0 -0.934172 -0.356822
v 0 -0.934172 0.356822
v 0.356822 0 0.934172
v -0.356822 0 0.934172
v 0.57735 0.57735 -0.57735
v 0.57735 0.57735 0.57735
v -0.57735 0.57735 -0.57735
v -0.57735 0.57735 0.57735
v 0.57735 -0.57735 -0.57735
v 0.57735 -0.57735 0.57735
v -0.57735 -0.57735 -0.57735
f 19 3 2
f 12 19 2
f 15 12 2
f 8 14 2
f 18 8 2
f 3 18 2
f 20 5 4
f 9 20 4
f 16 9 4
f 13 17 4
f 1 13 4
f 5 1 4
f 7 16 4
f 6 7 4
f 17 6 4
f 6 15 2
f 7 6 2
f 14 7 2
f 10 18 3
f 11 10 3
f 19 11 3
f 11 1 5
f 10 11 5
f 20 10 5
f 20 9 8
f 10 20 8
f 18 10 8
f 9 16 7
f 8 9 7
f 14 8 7
f 12 15 6
f 13 12 6
f 17 13 6
f 13 1 11
f 12 13 11
f 19 12 11
)";

std::string icosahedronObj = R"(v 0 -0.525731 0.850651
v 0.850651 0 0.525731
v 0.850651 0 -0.525731
v -0.850651 0 -0.525731
v -0.850651 0 0.525731
v -0.525731 0.850651 0
v 0.525731 0.850651 0
v 0.525731 -0.850651 0
v -0.525731 -0.850651 0
v 0 -0.525731 -0.850651
v 0 0.525731 -0.850651
v 0 0.525731 0.850651
f 2 3 7
f 2 8 3
f 4 5 6
f 5 4 9
f 7 6 12
f 6 7 11
f 10 11 3
f 11 10 4
f 8 9 10
f 9 8 1
f 12 1 2
f 1 12 5
f 7 3 11
f 2 7 12
f 4 6 11
f 6 5 12
f 3 8 10
f 8 2 1
f 4 10 9
f 5 9 1
)";

struct TrigFace {
    vec3 vtx[3];
    TrigFace(vec3 a, vec3 b, vec3 c){
        vtx[0] = a;
        vtx[1] = b;
        vtx[2] = c;
    }
    inline vec3& operator[](int i) { return vtx[i]; }
    inline vec3 operator[](int i) const { return vtx[i]; }
};

// cpu raytracer demo
struct Hit {
    float t;
    vec3 position, normal;
    Hit() { t = -1; }
};

// cpu raytracer demo
struct Ray {
    vec3 start, dir;
    Ray(vec3 _start, vec3 _dir) { start = _start; dir = normalize(_dir); }
};

// cpu raytracer demo
class Intersectable {
public:
    virtual Hit intersect(const Ray& ray) = 0;
};

class TriangularIntersectable : public Intersectable {
protected:
    vec3 pos; // world coord
    std::vector<TrigFace> faces;
    float scale;

    void loadObj(const std::string& objString){
        std::vector<vec3> vertices;
        size_t start = 0;
        size_t end = objString.find('\n', 0);
        std::string line;

        while (end != std::string::npos) {
            std::vector<std::string> splitArr;
            line = objString.substr(start, end-start);
            if (line.length() > 0){
                size_t lineStart = 0;
                size_t lineEnd = line.find(' ', 0);
                while (lineEnd != std::string::npos){
                    splitArr.push_back(line.substr(lineStart, lineEnd-lineStart));
                    lineStart = lineEnd + 1;
                    lineEnd = line.find(' ', lineStart);
                }
                if (lineStart < line.length())
                    splitArr.push_back(line.substr(lineStart));
            }
            start = end + 1;
            end = objString.find('\n', start);

            if (!splitArr.empty()){
                if (splitArr.at(0) == "v"){
                    vertices.push_back(vec3(
                        std::stof(splitArr.at(1)),
                        std::stof(splitArr.at(2)),
                        std::stof(splitArr.at(3))
                    ));
                } else if (splitArr.at(0) == "f"){
                    vec3 v1 = vertices.at(std::stoi(splitArr.at(1)) - 1);
                    vec3 v2 = vertices.at(std::stoi(splitArr.at(2)) - 1);
                    vec3 v3 = vertices.at(std::stoi(splitArr.at(3)) - 1);
                    faces.push_back(TrigFace(v1, v2, v3));
                }
            }
        }
    }
public:
    TriangularIntersectable(vec3 pos, float scale=1.0f) : pos(pos), scale(scale) {}

    Hit intersect(const Ray& ray) {
        Hit hit;
        for (size_t i=0; i<faces.size(); i++){
            // raytracing ea slide 12
            const vec3 r1 = pos + (scale * faces[i][0]);
            const vec3 r2 = pos + (scale * faces[i][1]);
            const vec3 r3 = pos + (scale * faces[i][2]);
            Hit tmp;
            tmp.normal = normalize(cross(r2-r1, r3-r1));

            tmp.t = dot(r1-ray.start, tmp.normal) / dot(ray.dir, tmp.normal);
            // skip if not hitting or greater distance
            if (tmp.t < 0 || (hit.t != -1 && tmp.t > hit.t)) continue;

            tmp.position = ray.start + (ray.dir * tmp.t);

            // ray hits inside triangle
            if (
                dot(cross(r2-r1, tmp.position-r1), tmp.normal) > 0 &&
                dot(cross(r3-r2, tmp.position-r2), tmp.normal) > 0 &&
                dot(cross(r1-r3, tmp.position-r3), tmp.normal) > 0
            ){
                hit = tmp;
            }
        }
        return hit;
    }
};

struct OpenCubeRoom : public TriangularIntersectable {
    OpenCubeRoom(vec3 pos) : TriangularIntersectable(pos) {
        loadObj(openCubeObj);
    }
};

struct Dodecahedron : public TriangularIntersectable {
    Dodecahedron(vec3 pos, float scale=.2f) : TriangularIntersectable(pos, scale) {
        loadObj(dodecahedronObj);
    }
};

struct Icosahedron : public TriangularIntersectable {
    Icosahedron(vec3 pos, float scale=.2f) : TriangularIntersectable(pos, scale) {
        loadObj(icosahedronObj);
    }
};

const float epsilon = 0.0001f;

struct IlluminatingCone : public Intersectable {
    float height = .1f;
    float alpha = 22.5f * M_PI / 180.0f;
    vec3 color;
    vec3 pos = vec3(0, 0, 0);
    vec3 normal = vec3(0, 0, 0);
    IlluminatingCone(vec3 color) : color(color){}

    Hit intersect(const Ray &ray) {
        // raytracing ea slide 11
        Hit hit;
        float cosa = cosf(alpha);

        float a = (dot(ray.dir, normal) * dot(ray.dir, normal))
            - (cosa * cosa * dot(ray.dir, ray.dir)
        );
        float b = 2 * ((dot(ray.dir, normal) * dot(ray.start - pos, normal))
            - (cosa * cosa * dot(ray.dir, ray.start - pos))
        );
        float c = (dot(ray.start - pos, normal) * dot(ray.start - pos, normal))
            - (cosa * cosa * dot(ray.start - pos, ray.start - pos)
        );

        float D = (b * b) - (4.0f * a * c);
        float t1 = (-b + sqrtf(D)) / (2.0f * a);
        float t2 = (-b - sqrtf(D)) / (2.0f * a);
        if (D < 0) return hit;

        float t;
        vec3 r;
        for (size_t i=0; i<3; i++){
            if (i == 2)
                return hit;

            t = ((i==0) ? t1 : t2);
            if (t < 0)
                continue;

            r = ray.start + ray.dir * ((i==0) ? t1 : t2);
            float lenCheck = dot(r - pos, normal);
            if (!(0 <= lenCheck && lenCheck <= height))
                continue;
            else
                break;
        }

        hit.t = t;
        hit.position = r;
        hit.normal = normalize (2.0f * ((dot(r - pos, normal) * normal) - ((r - pos) * cosa * cosa)));

        return hit;
    }
};

// cpu raytracer demo
class Camera {
    vec3 eye, lookat, right, up;
public:
    void set(vec3 _eye, vec3 _lookat, vec3 vup, float fov) {
        eye = _eye;
        lookat = _lookat;
        vec3 w = eye - lookat;
        float focus = length(w);
        right = normalize(cross(vup, w)) * focus * tanf(fov / 2);
        up = normalize(cross(w, right)) * focus * tanf(fov / 2);
    }
    Ray getRay(int X, int Y) {
        vec3 dir = lookat + right * (2.0f * (X + 0.5f) / windowWidth - 1) + up * (2.0f * (Y + 0.5f) / windowHeight - 1) - eye;
        return Ray(eye, dir);
    }
};


// uses code from cpu raytracer demo
class Scene {
    std::vector<Intersectable *> objects;
    std::vector<IlluminatingCone *> lightCones;
    Camera camera;
    vec3 bg = vec3(0, 0, 0);
public:
    void build() {
        vec3 eye = vec3(1.35f, 0, 1.2f), vup = vec3(0, 1, 0), lookat = vec3(0, 0, 0);
        float fov = 45 * M_PI / 180;
        camera.set(eye, lookat, vup, fov);

        objects.push_back(new OpenCubeRoom(vec3(-.6f, -.5f, -.5f)));
        objects.push_back(new Dodecahedron(vec3(0, -.3f, .3f)));
        objects.push_back(new Icosahedron(vec3(.3f, -.3f, -.2f)));

        lightCones.push_back(new IlluminatingCone(vec3(1.0f, 0, 0)));
        objects.push_back(lightCones.at(0));
        lightCones.push_back(new IlluminatingCone(vec3(0, 1.0f, 0)));
        objects.push_back(lightCones.at(1));
        lightCones.push_back(new IlluminatingCone(vec3(0, 0, 1.0f)));
        objects.push_back(lightCones.at(2));

        updateConePosition(lightCones.at(0), .44*windowWidth, .5*windowHeight);
        updateConePosition(lightCones.at(1), .65*windowWidth, .85*windowHeight);
        updateConePosition(lightCones.at(2), .38*windowWidth, .4*windowHeight);
    }

void render(std::vector<vec4>& image) {
#pragma omp parallel for
        for (unsigned int Y = 0; Y < windowHeight; Y++) {
//#pragma omp parallel for
            for (unsigned int X = 0; X < windowWidth; X++) {
                vec3 color = trace(camera.getRay(X, Y));
                image[Y * windowWidth + X] = vec4(color.x, color.y, color.z, 1);
            }
        }
    }

    Hit firstIntersect(Ray ray, bool viewInvert=true, bool notCone=false) {
        Hit bestHit;
        for (Intersectable * object : objects) {
            if (notCone)
                for (size_t i=0; i<lightCones.size(); i++)
                    if (lightCones.at(i) == object)
                        continue;

            Hit hit = object->intersect(ray); //  hit.t < 0 if no intersection
            if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))  bestHit = hit;
        }
        if (viewInvert && bestHit.t > 0 && dot(ray.dir, bestHit.normal) > 0)
            bestHit.normal = bestHit.normal * (-1);

        return bestHit;
    }

    bool shadowIntersect(Ray ray) {	// for directional lights
        for (Intersectable * object : objects) if (object->intersect(ray).t > 0) return true;
        return false;
    }

    void updateConePosition(IlluminatingCone* cone, int pX, int pY){// window coord
        Hit hit = firstIntersect(camera.getRay(pX, pY), true, true);
        if (hit.t > 0){
            cone->pos = hit.position;
            cone->normal = normalize(hit.normal);
        }
    }

    vec3 lightConeColorIntensity(IlluminatingCone* cone, Hit target){
        vec3 colorIntensity(0, 0, 0);
        vec3 conePtoTargetPosDir = normalize(target.position - cone->pos);

        // too big angle
        if (dot(cone->normal, conePtoTargetPosDir) < cosf(cone->alpha) + epsilon)
            return colorIntensity;

        Hit intersect = firstIntersect(Ray(cone->pos + (cone->normal * epsilon), conePtoTargetPosDir));
        if (intersect.t < 0
        || fabs(intersect.t - length(target.position - (cone->pos + (cone->normal * epsilon)))) > /*50.0f **/ epsilon)
            return colorIntensity;

        colorIntensity = (.5f / (intersect.t * intersect.t + .5f)) * cone->color;
        return colorIntensity;
    }

    IlluminatingCone* getClosestCone(int pX, int pY){
        Hit hit = firstIntersect(camera.getRay(pX, pY), true, true);
        IlluminatingCone* result = nullptr;
        if (hit.t > 0){
            for (IlluminatingCone* cone: lightCones)
                if (result == nullptr || length(hit.position - cone->pos) < length(hit.position - result->pos))
                    result = cone;
        }
        return result;
    }

    vec3 trace(Ray ray, int depth = 0) {
        Hit hit = firstIntersect(ray);
        if (hit.t < 0) return bg;

        float cosDelta = dot(hit.normal, -ray.dir);
        float baseColor = .2f * (1 + cosDelta);
        vec3 outColor(baseColor, baseColor, baseColor);

        for (IlluminatingCone* cone : lightCones){
            outColor = outColor + lightConeColorIntensity(cone, hit);
        }

        return outColor;
    }
};

// cpu raytracer demo
// vertex shader in GLSL: It is a Raw string (C++11) since it contains new line characters
const char *vertexSource = R"(
	#version 330
    precision highp float;

	layout(location = 0) in vec2 cVertexPosition;	// Attrib Array 0
	out vec2 texcoord;

	void main() {
		texcoord = (cVertexPosition + vec2(1, 1))/2;							// -1,1 to 0,1
		gl_Position = vec4(cVertexPosition.x, cVertexPosition.y, 0, 1); 		// transform to clipping space
	}
)";

// cpu raytracer demo
// fragment shader in GLSL
const char *fragmentSource = R"(
	#version 330
    precision highp float;

	uniform sampler2D textureUnit;
	in  vec2 texcoord;			// interpolated texture coordinates
	out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation

	void main() {
		fragmentColor = texture(textureUnit, texcoord);
	}
)";

GPUProgram gpuProgram; // vertex and fragment shaders
Scene scene;
//unsigned int vao;	   // virtual world on the GPU

// cpu raytracer demo
class FullScreenTexturedQuad {
    unsigned int vao;	// vertex array object id and texture id
    Texture texture;
public:
    FullScreenTexturedQuad(/*int windowWidth, int windowHeight, std::vector<vec4>& image*/)
            //: texture(windowWidth, windowHeight, image)
    {
        glGenVertexArrays(1, &vao);	// create 1 vertex array object
        glBindVertexArray(vao);		// make it active

        unsigned int vbo;		// vertex buffer objects
        glGenBuffers(1, &vbo);	// Generate 1 vertex buffer objects

        // vertex coordinates: vbo0 -> Attrib Array 0 -> vertexPosition of the vertex shader
        glBindBuffer(GL_ARRAY_BUFFER, vbo); // make it active, it is an array
        float vertexCoords[] = { -1, -1,  1, -1,  1, 1,  -1, 1 };	// two triangles forming a quad
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_DYNAMIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);     // stride and offset: it is tightly packed
    }

    void setImage(const std::vector<vec4>& im){
        texture.create(windowWidth, windowHeight, im);
    }

    void Draw() {
        glBindVertexArray(vao);	// make the vao and its vbos active playing the role of the data source
        gpuProgram.setUniform(texture, "textureUnit");
        glDrawArrays(GL_TRIANGLE_FAN, 0, 4);	// draw two triangles forming a quad
    }
};
FullScreenTexturedQuad* fullScreenTexturedQuad = nullptr;

void renderScene(){
    long timeStart = glutGet(GLUT_ELAPSED_TIME);
    std::vector<vec4> image(windowWidth * windowHeight);
    scene.render(image);
    long timeEnd = glutGet(GLUT_ELAPSED_TIME);
    printf("Rendering time: %d milliseconds\n", (timeEnd - timeStart));

    if (fullScreenTexturedQuad == nullptr)
        fullScreenTexturedQuad = new FullScreenTexturedQuad();

    fullScreenTexturedQuad->setImage(image);
}

// Initialization, create an OpenGL context
void onInitialization() {
    glViewport(0, 0, windowWidth, windowHeight);
    scene.build();

    renderScene();
    gpuProgram.create(vertexSource, fragmentSource, "fragmentColor");
}

// Window has become invalid: Redraw
void onDisplay() {
    fullScreenTexturedQuad->Draw();
    glutSwapBuffers();
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
    if (key == 'd') glutPostRedisplay();
    if (key == 'r') { renderScene(); glutPostRedisplay(); }
//    if (key == 'q') glutExit();
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {	// pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) { // pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
    pY = windowHeight - pY;

	if (state == GLUT_DOWN && button == GLUT_LEFT_BUTTON) {
        IlluminatingCone *cone = scene.getClosestCone(pX, pY);
        if (cone != nullptr) {
            scene.updateConePosition(cone, pX, pY);
            renderScene();
            glutPostRedisplay();
        }
    }
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
//	long time = glutGet(GLUT_ELAPSED_TIME); // elapsed time since the start of the program
}
