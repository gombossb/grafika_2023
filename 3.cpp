//=============================================================================================
// Mintaprogram: Zöld háromszög. Ervenyes 2019. osztol.
// + Computer Graphics Sample Program: 3D engine-let
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

const int tessellationLevel = 70;
const float g = 0.08;

// source: 3d engine-let
struct Camera { // 3D camera
    vec3 wEye, wLookat, wVup;   // extrinsic
    float fov, asp, fp, bp;		// intrinsic
public:
    Camera() {
        asp = (float)windowWidth / 2.0f / windowHeight; // half width camera
        fov = 75.0f * (float)M_PI / 180.0f;
        fp = .1f; bp = 30;
    }
    mat4 V() { // view matrix: translates the center to the origin
        vec3 w = normalize(wEye - wLookat);
        vec3 u = normalize(cross(wVup, w));
        vec3 v = cross(w, u);
        return TranslateMatrix(wEye * (-1)) * mat4(u.x, v.x, w.x, 0,
                                                   u.y, v.y, w.y, 0,
                                                   u.z, v.z, w.z, 0,
                                                   0,   0,   0,   1);
    }

    mat4 P() { // projection matrix
        return mat4(1 / (tan(fov / 2)*asp), 0,                0,                      0,
                    0,                      1 / tan(fov / 2), 0,                      0,
                    0,                      0,                -(fp + bp) / (bp - fp), -1,
                    0,                      0,                -2 * fp*bp / (bp - fp),  0);
    }
};

// source: 3d engine-let
struct Material {
    vec3 kd, ks, ka;
    float shininess;
};

// source: 3d engine-let
struct Light {
    vec3 La, Le;
    vec4 wLightPos; // homogeneous coordinates, can be at ideal point
};

// source: 3d engine-let
struct RenderState {
    mat4	           MVP, M, Minv, V, P;
    Material *         material;
    std::vector<Light> lights;
    vec3	           wCam;
};

// source: 3d engine-let
class Shader : public GPUProgram {
public:
    virtual void Bind(RenderState state) = 0;

    void setUniformMaterial(const Material& material, const std::string& name) {
        setUniform(material.kd, name + ".kd");
        setUniform(material.ks, name + ".ks");
        setUniform(material.ka, name + ".ka");
        setUniform(material.shininess, name + ".shininess");
    }

    void setUniformLight(const Light& light, const std::string& name) {
        setUniform(light.La, name + ".La");
        setUniform(light.Le, name + ".Le");
        setUniform(light.wLightPos, name + ".wLightPos");
    }
};

// source: 3d engine-let
class PhongShader : public Shader {
    const char * vertexSource = R"(
		#version 330
		precision highp float;

		struct Light {
			vec3 La, Le;
			vec4 wLightPos;
		};

		uniform mat4  MVP, M, Minv; // MVP, Model, Model-inverse
		uniform Light[8] lights;    // light sources
		uniform int   nLights;
		uniform vec3  wCam;         // pos of eye

		layout(location = 0) in vec3  vtxPos;            // pos in modeling space
		layout(location = 1) in vec3  vtxNorm;      	 // normal in modeling space
		layout(location = 2) in vec3  vtxColor;

		out vec3 wNormal;		    // normal in world space
		out vec3 wView;             // view in world space
		out vec3 wLight[8];		    // light dir in world space
		out vec3 color;

		void main() {
			gl_Position = vec4(vtxPos, 1) * MVP; // to NDC

			vec4 wPos = vec4(vtxPos, 1) * M;
			for(int i = 0; i < nLights; i++) {
				wLight[i] = lights[i].wLightPos.xyz * wPos.w - wPos.xyz * lights[i].wLightPos.w;
			}
		    wView  = wCam * wPos.w - wPos.xyz;
		    wNormal = (Minv * vec4(vtxNorm, 0)).xyz;
            color = vtxColor;
		}
	)";

    const char * fragmentSource = R"(
		#version 330
		precision highp float;

		struct Light {
			vec3 La, Le;
			vec4 wLightPos;
		};

		struct Material {
			vec3 kd, ks, ka;
			float shininess;
		};

		uniform Material material;
		uniform Light[8] lights;    // light sources
		uniform int   nLights;

		in  vec3 wNormal;       // interpolated world sp normal
		in  vec3 wView;         // interpolated world sp view
		in  vec3 wLight[8];     // interpolated world sp illum dir
        in  vec3 color;

        out vec4 fragmentColor; // output goes to frame buffer

		void main() {
			vec3 N = normalize(wNormal);
			vec3 V = normalize(wView);
			if (dot(N, V) < 0) N = -N;	// prepare for one-sided surfaces like Mobius or Klein

			vec3 ka = material.ka * color;
//			vec3 kd = material.kd * color;

			vec3 radiance = vec3(0, 0, 0);
			for(int i = 0; i < nLights; i++) {
				vec3 L = normalize(wLight[i]);
				vec3 H = normalize(L + V);
				float cost = max(dot(N,L), 0), cosd = max(dot(N,H), 0);

				radiance += (ka * lights[i].La) + (color * cost) + (material.ks * pow(cosd, material.shininess));
			}
			fragmentColor = vec4(radiance, 1);
		}
	)";
public:
    PhongShader() { create(vertexSource, fragmentSource, "fragmentColor"); }

    void Bind(RenderState state) {
        Use();
        setUniform(state.MVP, "MVP");
        setUniform(state.M, "M");
        setUniform(state.Minv, "Minv");
        setUniform(state.wCam, "wCam");
        setUniformMaterial(*state.material, "material");

        setUniform((int)state.lights.size(), "nLights");
        for (unsigned int i = 0; i < state.lights.size(); i++) {
            setUniformLight(state.lights[i], std::string("lights[") + std::to_string(i) + std::string("]"));
        }
    }
};

// source: 3d engine-let
class Geometry {
protected:
    unsigned int vao, vbo;
public:
    Geometry() {
        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);
        glGenBuffers(1, &vbo);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
    }
    virtual void Draw() = 0;
    ~Geometry() {
        glDeleteBuffers(1, &vbo);
        glDeleteVertexArrays(1, &vao);
    }
};

struct VertexData {
    vec3 position, normal;
    vec3 color;
    VertexData(const vec3 &position, const vec3 &normal, const vec3 &color)
        : position(position), normal(normal), color(color) {}
    VertexData(){}
};

// source: 3d engine-let
class ParamSurface : public Geometry {
    unsigned int nVtxPerStrip = 0, nStrips = 0;
public:
    virtual VertexData GenVertexData(float u, float v, float phi) = 0;

    void create(int N = tessellationLevel, int M = tessellationLevel) {
        nVtxPerStrip = (M + 1) * 2;
        nStrips = N;
        std::vector<VertexData> vtxData;
        for (int i = 0; i < N; i++) {
            for (int j = 0; j <= M; j++) {
                float phi = (rand() / RAND_MAX) * M_PI * i;
                vtxData.push_back(GenVertexData((float)j / M, (float)i / N, phi));
                vtxData.push_back(GenVertexData((float)j / M, (float)(i + 1) / N, phi));
            }
        }
        glBufferData(GL_ARRAY_BUFFER, nVtxPerStrip * nStrips * sizeof(VertexData), &vtxData[0], GL_STATIC_DRAW);
        // Enable the vertex attribute arrays
        glEnableVertexAttribArray(0);  // attribute array 0 = POSITION
        glEnableVertexAttribArray(1);  // attribute array 1 = NORMAL
        glEnableVertexAttribArray(2);  // attribute array 2 = HEIGHT
        // attribute array, components/attribute, component type, normalize?, stride, offset
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, position));
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, normal));
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, color));
    }

    void Draw() {
        glBindVertexArray(vao);
        for (unsigned int i = 0; i < nStrips; i++)
            glDrawArrays(GL_TRIANGLE_STRIP, i *  nVtxPerStrip, nVtxPerStrip);
    }
};

class NoiseLand : public ParamSurface {
    const unsigned int n = 4;
    const float A0 = .15f;
    float A(float f1, float f2){
        if (f1+f2 > 0)
            return A0 / sqrtf(f1*f1 + f2*f2);

        return 0;
    }
public:
    float minH=0, maxH=0;
    NoiseLand() { create(); printf("%g,%g\n", minH, maxH); }
    VertexData GenVertexData(float u, float v, float phi){
        VertexData vd;
        float x = 2 * M_PI * u - M_PI;
        float y = 2 * M_PI * v - M_PI;

        float h=0, h_x=0, h_y=0;
        for (int f1 = 0; f1 <= n; f1+=1){
            for (int f2 = 0; f2 <= n; f2+=1){
                h += A(f1, f2) * cosf(f1 * x + f2 * y + phi);
                h_x += A(f1, f2) * x * sinf(f1 * x + f2 * y + phi);
                h_y += A(f1, f2) * y * sinf(f1 * x + f2 * y + phi);
            }
        }
        if (h < minH) minH = h;
        if (h > maxH) maxH = h;
        vd.position = vec3(x, y, h);
        vd.normal = normalize(vec3(-h_x, -h_y, 1));

        vec3 green(0.0, .8, 0.0);
        vec3 brown(.13, .09, .01);

        float normalizedH = (h/1.5) + .2;

        vd.color = normalizedH * ((1-normalizedH) * green + normalizedH * brown);

        return vd;
    }
};

struct BungeeBoxGeom : public Geometry {
    BungeeBoxGeom(vec3 color){
        std::vector<VertexData> vtxData;

        // front face
        vtxData.push_back(VertexData(vec3(0, 0, 1), vec3(0, -1, 0), color));
        vtxData.push_back(VertexData(vec3(0, 0, 0), vec3(0, -1, 0), color));
        vtxData.push_back(VertexData(vec3(1, 0, 1), vec3(0, -1, 0), color));
        vtxData.push_back(VertexData(vec3(1, 0, 0), vec3(0, -1, 0), color));

        // right face
        vtxData.push_back(VertexData(vec3(1, 1, 0), vec3(1, 0, 0), color));
        vtxData.push_back(VertexData(vec3(1, 0, 0), vec3(1, 0, 0), color));
        vtxData.push_back(VertexData(vec3(1, 1, 1), vec3(1, 0, 0), color));
        vtxData.push_back(VertexData(vec3(1, 0, 1), vec3(1, 0, 0), color));

        // top face
        vtxData.push_back(VertexData(vec3(1, 0, 1), vec3(0, 0, 1), color));
        vtxData.push_back(VertexData(vec3(0, 0, 1), vec3(0, 0, 1), color));
        vtxData.push_back(VertexData(vec3(1, 1, 1), vec3(0, 0, 1), color));
        vtxData.push_back(VertexData(vec3(0, 1, 1), vec3(0, 0, 1), color));

        // left face
        vtxData.push_back(VertexData(vec3(0, 0, 1), vec3(-1, 0, 0), color));
        vtxData.push_back(VertexData(vec3(0, 0, 0), vec3(-1, 0, 0), color));
        vtxData.push_back(VertexData(vec3(0, 1, 1), vec3(-1, 0, 0), color));
        vtxData.push_back(VertexData(vec3(0, 1, 0), vec3(-1, 0, 0), color));

        // bottom face
        vtxData.push_back(VertexData(vec3(0, 0, 0), vec3(0, 0, -1), color));
        vtxData.push_back(VertexData(vec3(1, 0, 0), vec3(0, 0, -1), color));
        vtxData.push_back(VertexData(vec3(0, 1, 0), vec3(0, 0, -1), color));
        vtxData.push_back(VertexData(vec3(1, 1, 0), vec3(0, 0, -1), color));

        // rear face
        vtxData.push_back(VertexData(vec3(1, 1, 0), vec3(0, 1, 0), color));
        vtxData.push_back(VertexData(vec3(0, 1, 0), vec3(0, 1, 0), color));
        vtxData.push_back(VertexData(vec3(1, 1, 1), vec3(0, 1, 0), color));
        vtxData.push_back(VertexData(vec3(0, 1, 1), vec3(0, 1, 0), color));

        glBufferData(GL_ARRAY_BUFFER, 6 * 4 * sizeof(VertexData), &vtxData[0], GL_STATIC_DRAW);
        // Enable the vertex attribute arrays
        glEnableVertexAttribArray(0);  // attribute array 0 = POSITION
        glEnableVertexAttribArray(1);  // attribute array 1 = NORMAL
        glEnableVertexAttribArray(2);  // attribute array 2 = COLOR
        // attribute array, components/attribute, component type, normalize?, stride, offset
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, position));
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, normal));
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, color));
    }

    void Draw(){
        glBindVertexArray(vao);
        for (int i = 0; i < 6; i++)
            glDrawArrays(GL_TRIANGLE_STRIP, i * 4, 4);
    }
};

// source: 3d engine-let
struct Object {
    Shader *   shader;
    Material * material;
    Geometry * geometry;
    vec3 scale, translation, rotationAxis;
    float rotationAngle;
public:
    Object(Shader * _shader, Material * _material, Geometry * _geometry)
    : scale(vec3(1, 1, 1)), translation(vec3(0, 0, 0)), rotationAxis(0, 0, 1), rotationAngle(0) {
        shader = _shader;
        material = _material;
        geometry = _geometry;
    }

    virtual void SetModelingTransform(mat4& M, mat4& Minv) {
        M = ScaleMatrix(scale) * RotationMatrix(rotationAngle, rotationAxis) * TranslateMatrix(translation);
        Minv = TranslateMatrix(-translation) * RotationMatrix(-rotationAngle, rotationAxis)
            * ScaleMatrix(vec3(1 / scale.x, 1 / scale.y, 1 / scale.z));
    }

    void Draw(RenderState state) {
        mat4 M, Minv;
        SetModelingTransform(M, Minv);
        state.M = M;
        state.Minv = Minv;
        state.MVP = state.M * state.V * state.P;
        state.material = material;
        shader->Bind(state);
        geometry->Draw();
    }
};

struct BungeeBox : public Object {
    Camera cam;
    const float a = .2f, b = .3f, c = .5f;
    const float m = 1, D = .5, L0 = .2;
    const float A = 45 * M_PI / 180.0;
    BungeeBox(Shader *shader, Material *material, vec3 color)
    : Object(shader, material, new BungeeBoxGeom(color)) {
        scale = vec3(a, b, c);
        translation = vec3(-a/2, -b/2, 1.9);
        rotationAxis = vec3(1, 0, 0);

        cam.wVup = vec3(0, 1, 0);
    }
    void updatePos(float t){
        rotationAngle = A * sinf(2 * t);
        cam.wEye = translation + rotationAxis * rotationAngle;
        cam.wLookat = vec3(translation.x, 10 * rotationAngle, -5);
    }
};

class Scene {
    std::vector<Object *> objects;
    std::vector<Light> lights;
public:
    Camera droneCam;
    BungeeBox *box;
    void Build() {
        Shader * phongShader = new PhongShader();

        Material * landMat = new Material;
        landMat->kd = vec3(0.0f, 0.0f, 0.0f);
        landMat->ks = vec3(.3f, .3f, .3f);
        landMat->ka = vec3(0.0f, 0.0f, 0.0f);
        landMat->shininess = 20;

        Geometry * noiseLand = new NoiseLand();

        Object * noiseObject1 = new Object(phongShader, landMat, noiseLand);
        noiseObject1->translation = vec3(0, 0, 0);
        noiseObject1->scale = vec3(1, 1, 1);
        objects.push_back(noiseObject1);

        Material * boxMat = new Material;
        boxMat->kd = vec3(0.3f, 0.3f, 0.2f);
        boxMat->ks = vec3(.3f, .3f, .3f);
        boxMat->ka = vec3(0.1f, 0.1f, 0.1f);
        boxMat->shininess = 40;

        box = new BungeeBox(phongShader, boxMat, vec3(.5, 0, .5));
        objects.push_back(box);

        droneCam.wLookat = vec3(0, 0, 0);
        droneCam.wVup = vec3(0, 0, 1);

        lights.resize(4);
        lights[0].wLightPos = vec4(5, 5, 4, 0);
        lights[0].La = vec3(0.1f, 0.1f, 1);
        lights[0].Le = vec3(.7, .6, .8);

        lights[1].wLightPos = vec4(5, -5, 10, 0);
        lights[1].La = vec3(0.2f, 0.2f, 0.2f);
        lights[1].Le = vec3(.2, 1, .3);

        lights[2].wLightPos = vec4(-5, -5, 3, 0);
        lights[2].La = vec3(0.1f, 0.1f, 0.1f);
        lights[2].Le = vec3(.5, .6, 1);

        lights[3].wLightPos = vec4(-5, 5, 7, 0);
        lights[3].La = vec3(0.1f, 0.1f, 0.1f);
        lights[3].Le = vec3(.5, .6, 1);
    }

    void RenderDroneCam() {
        RenderState state;
        state.wCam = droneCam.wEye;
        state.V = droneCam.V();
        state.P = droneCam.P();
        state.lights = lights;
        for (Object * obj : objects)
            obj->Draw(state);
    }
    void RenderBoxCam() {
        RenderState state;
        state.wCam = box->cam.wEye;
        state.V = box->cam.V();
        state.P = box->cam.P();
        state.lights = lights;
        for (Object * obj : objects)
            obj->Draw(state);
    }
};

Scene scene;

void onInitialization() {
    glEnable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);
    scene.Build();
}

void onDisplay() {
    glViewport(0, 0, windowWidth, windowHeight);
    glClearColor(0.5f, 0.5f, 0.8f, 1.0f); // background color
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear the screen


    glViewport(0, 0, windowWidth/2, windowHeight);
    scene.RenderBoxCam();

    glViewport(windowWidth/2, 0, windowWidth/2, windowHeight);
    scene.RenderDroneCam();

    glutSwapBuffers();
}

void onKeyboardUp(unsigned char key, int pX, int pY) { }

void onMouse(int button, int state, int pX, int pY) { }

void onMouseMotion(int pX, int pY) {
}

float lastFrame = 0;
float tSinceJump = 0;

void onKeyboard(unsigned char key, int pX, int pY) {
    tSinceJump = 0;
}

void onIdle() {
    static float t = 0;
    t = glutGet(GLUT_ELAPSED_TIME) / 1000.0f;
    const float dt = lastFrame - t;
    tSinceJump += dt;
    lastFrame = t;

    scene.box->updatePos(tSinceJump);
    scene.droneCam.wEye = vec3(1.2*M_PI*cosf(t), 1.2 * M_PI * sinf(t), 4.0f);
    glutPostRedisplay();
}
