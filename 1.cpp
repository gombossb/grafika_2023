//=============================================================================================
// Mintaprogram: Zöld háromszög. Ervenyes 2019. osztol.
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

// vertex shader in GLSL: It is a Raw string (C++11) since it contains new line characters
const char * const vertexSource = R"(
	#version 330				// Shader 3.3
	precision highp float;		// normal floats, makes no difference on desktop computers

	//uniform mat4 MVP;			// uniform variable, the Model-View-Projection transformation matrix
	layout(location = 0) in vec2 vp;	// Varying input: vp = vertex position is expected in attrib array 0

	void main() {
		gl_Position = vec4(vp.x, vp.y, 0, 1)/* * MVP*/;		// transform vp from modeling space to normalized device space
	}
)";

// fragment shader in GLSL
const char * const fragmentSource = R"(
	#version 330			// Shader 3.3
	precision highp float;	// normal floats, makes no difference on desktop computers
	
	uniform vec3 color;		// uniform variable, the color of the primitive
	out vec4 outColor;		// computed color of the current pixel

	void main() {
		outColor = vec4(color, 1);	// computed color is the color of the primitive
	}
)";

GPUProgram gpuProgram; // vertex and fragment shaders

inline float hypDot(vec3 a, vec3 b){
    return a.x * b.x + a.y * b.y - a.z*b.z; // w->z, z >= 0
}

inline float hypLength(vec3 v){
    return sqrtf(hypDot(v, v));
}
inline vec3 hypNormalize(vec3 v) {
    return v * (1 / hypLength(v));
}

inline bool isValidHypPos(vec3 p){
    return p.z >= 0 && fabs(hypDot(p, p) - (-1)) < 0.0001f;
}

inline float hypCalcWForXY(vec2 p){
    return sqrtf(p.x*p.x + p.y*p.y + 1);
}
// against float's inaccuracy
inline vec3 hypWFixer(vec3 p){
    return vec3(p.x, p.y, hypCalcWForXY(vec2(p.x, p.y)));
}

// w->z
// p: point on hyperboloid
// returns 2d coord mapped to z=0 plane
inline vec2 hyp2DPoincareProjection(vec3 p){
    return vec2(p.x/(p.z+1), p.y/(p.z+1));
}

// 1. Egy irányra mer?leges irány állítása.
inline vec3 hypOrthogonal(vec3 p, vec3 v){
    vec3 p_ = vec3(p.x, p.y, -1.0f*p.z);
    vec3 v_ = vec3(v.x, v.y, -1.0f*v.z);
    return cross(p_, v_);
}

// 2. Adott pontból és sebesség vektorral induló pont helyének és sebesség
// vektorának számítása t id?vel kés?bb.
inline vec3 hypLerpPos(vec3 p, vec3 v, float t){
    return (p * coshf(t)) + (hypNormalize(v) * sinhf(t)); // EA1 slide 81
}
// deriv of hypLerpPos
inline vec3 hypLerpVel(vec3 p, vec3 v, float t){
    return (p * sinhf(t)) + (hypNormalize(v) * coshf(t));
}

// 3. Egy ponthoz képest egy másik pont irányának és távolságának meghatározása.
inline float hypDistance(vec3 p, vec3 q){
    return acoshf(-hypDot(p, q));
}
inline vec3 hypNormalizedDiffDir(vec3 p, vec3 q){
    vec3 diff = q - p;
    return hypNormalize(diff);
}

inline vec3 hypLerp(vec3 p, vec3 q, float t){
    float d = hypDistance(p, q);
    return p * (sinhf((1-t) * d) / sinhf(d)) + q * (sinhf(t*d) / sinhf(d));
}

// 4. Egy ponthoz képest adott irányban és távolságra lév? pont el?állítása.
inline vec3 hypFarPointFromP(vec3 p, vec3 v0, float d){
    return hypLerpPos(p, hypNormalize(v0), d);
}

// 5. Egy pontban egy vektor elforgatása adott szöggel.
inline vec3 hypRotate(vec3 p, vec3 v, float phi){ // v is normalized
    return (hypNormalize(v) * cosf(phi)) + (hypOrthogonal(p, hypNormalize(v)) * sinf(phi));
}

// 6. Egy közelít? pont és sebességvektorhoz a geometria szabályait teljesít?,
// közeli pont és sebesség választása.
// partial deriv of x^2 + y^2 - w^2 = -1, point as vector
inline vec3 hypTangentOfPoint(vec3 p){
    return vec3(2*p.x, 2*p.y, 2*(p.z-1.0f));
}

const size_t nCircleVertices = 30;
struct HypCircle {
    unsigned int vao, vbo;
    vec3 pos;
    vec3 dir;
    float radius;
    vec3 color;
    vec2 circleVertices[nCircleVertices] = {};

    HypCircle(vec3 pos, vec3 dir, float radius, vec3 color)
    : pos(pos), dir(hypNormalize(dir)), radius(radius), color(color) {
        glGenVertexArrays(1, &vao); glBindVertexArray(vao);
        glGenBuffers(1, &vbo); glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
        calcVertices();
    }
    void calcVertices(){
        for (size_t i = 0; i < nCircleVertices; i++){
            float phi = ((float)i) * 2.0f * M_PI / nCircleVertices;
            circleVertices[i] = hyp2DPoincareProjection(
                hypFarPointFromP(pos, hypRotate(pos, dir, phi), radius)
            );
        }
    }
    void updateGPU(){
        calcVertices();
        glBindVertexArray(vao); glBindBuffer(GL_ARRAY_BUFFER,vbo);
        glBufferData(GL_ARRAY_BUFFER, nCircleVertices * sizeof(vec2),
             circleVertices, GL_DYNAMIC_DRAW);
    }
    void draw(){
        glBindVertexArray(vao);
        gpuProgram.setUniform(color, "color");
        glDrawArrays(GL_TRIANGLE_FAN, 0, nCircleVertices);
    }
};

struct HypTrail {
    unsigned int vao, vbo;
    const vec3 color = vec3(1.0f, 1.0f, 1.0f);
    std::vector<vec2> lineVertices;
    void addPoint(vec3 p){
        lineVertices.push_back(
            hyp2DPoincareProjection(p)
        );
    }
    HypTrail(){
        glGenVertexArrays(1, &vao); glBindVertexArray(vao);
        glGenBuffers(1, &vbo); glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
    }
    void updateGPU(){
        glBindVertexArray(vao); glBindBuffer(GL_ARRAY_BUFFER,vbo);
        glBufferData(GL_ARRAY_BUFFER, lineVertices.size() * sizeof(vec2),
            &lineVertices[0], GL_DYNAMIC_DRAW);
    }
    void draw(){
        if (lineVertices.size() > 0) {
            glBindVertexArray(vao);
            gpuProgram.setUniform(color, "color");
            glDrawArrays(  GL_LINE_STRIP, 0, lineVertices.size());
        }
    }
    void reset(){
        lineVertices = std::vector<vec2>();
    }
};
struct HamiBase {
    const float radius = .30f;
    const float eyeBgRadius = .077f;
    const float eyeRadius = .04f;
    const float mouthRadiusAmplitude = .05f;
    const float eyeMouthDist = 2.0f*radius*M_PI*25.0f/360.0f;

    vec3 pos, dir;

    HypCircle baseCircle, leftEyeBgCircle, leftEyeCircle,
        rightEyeBgCircle, rightEyeCircle, mouthCircle;
    vec3* otherEyePos = nullptr;

    HypTrail trail;
    bool trailEnabled;

    HamiBase(vec3 pos, vec3 dir, vec3 color, bool trailEnabled = true)
    : pos(pos), dir(hypNormalize(dir)),
      baseCircle(HypCircle(pos, hypNormalize(dir), radius, color)),
      leftEyeBgCircle(HypCircle(
          vec3(0, 0, 0),
          vec3(0, 0, 0),
          eyeBgRadius, vec3(1.0f, 1.0f, 1.0f)
          )
      ),
      leftEyeCircle(HypCircle(
          vec3(0, 0, 0),
          vec3(0, 0, 0),
          eyeRadius, vec3(0, 0, 1.0f)
          )
      ),
      rightEyeBgCircle(HypCircle(
          vec3(0, 0, 0),
          vec3(0, 0, 0),
          eyeBgRadius, vec3(1.0f, 1.0f, 1.0f)
          )
      ),
      rightEyeCircle(HypCircle(
        vec3(0, 0, 0),
        vec3(0, 0, 0),
        eyeRadius, vec3(0, 0, 1.0f)
        )
      ),
      mouthCircle(HypCircle(
              vec3(0, 0, 0),
              vec3(0, 0, 0),
              mouthRadiusAmplitude, vec3(0, 0, 0)
          )
      ),
      trailEnabled(trailEnabled)
    {
        updatePosition(pos, dir);
    }
    void calcMouth(){
        mouthCircle.pos = hypLerpPos(pos, dir, radius*.85f);
        mouthCircle.dir = hypLerpVel(pos, dir, radius*.85f);
    }
    void pulsateMouth(float t){
        mouthCircle.radius = (.25f * sinf(8.0f * t) + .75f) * mouthRadiusAmplitude;
        mouthCircle.updateGPU();
    }
    void calcEyeBg(){
        leftEyeBgCircle.pos = hypLerpPos(
            mouthCircle.pos,
            hypRotate(mouthCircle.pos, mouthCircle.dir, -M_PI/2.0f),
            eyeMouthDist
        );
        leftEyeBgCircle.dir = hypLerpVel(
            mouthCircle.pos,
            hypRotate(mouthCircle.pos, mouthCircle.dir, -M_PI/2.0f),
            eyeMouthDist
        );

        rightEyeBgCircle.pos = hypLerpPos(
            mouthCircle.pos,
            hypRotate(mouthCircle.pos, mouthCircle.dir, M_PI/2.0f),
            eyeMouthDist
        );
        rightEyeBgCircle.dir = hypLerpVel(
            mouthCircle.pos,
            hypRotate(mouthCircle.pos, mouthCircle.dir, M_PI/2.0f),
            eyeMouthDist
        );
    }
    void calcEye(){
        if (otherEyePos != nullptr){
            vec3 leftEyeDir = hypNormalizedDiffDir(leftEyeBgCircle.pos, *otherEyePos);
            leftEyeCircle.pos = hypFarPointFromP(leftEyeBgCircle.pos, leftEyeDir, eyeRadius * .8f);
            leftEyeCircle.pos = hypWFixer(leftEyeCircle.pos);
            leftEyeCircle.dir = hypNormalize(hypTangentOfPoint(leftEyeCircle.pos));

            vec3 rightEyeDir = hypNormalizedDiffDir(rightEyeBgCircle.pos, *otherEyePos);
            rightEyeCircle.pos = hypFarPointFromP(rightEyeBgCircle.pos, rightEyeDir, eyeRadius * .8f);
            rightEyeCircle.pos = hypWFixer(rightEyeCircle.pos);
            rightEyeCircle.dir = hypNormalize(hypTangentOfPoint(rightEyeCircle.pos));
        }
    }
    void updatePosition(vec3 p, vec3 d){
        pos = p;
        dir = d;
        baseCircle.pos = p;
        baseCircle.dir = d;

        calcMouth();
        calcEyeBg();

        updateGPU();
    }
    void moveForward(float amount){
        vec3 newPos = hypLerpPos(pos, dir, amount);
        vec3 fixedNewPos = hypWFixer(newPos);
        vec3 newDir = hypNormalize(hypLerpVel(pos, dir, amount));

        updatePosition(fixedNewPos, newDir);
        if (trailEnabled)
            trail.addPoint(newPos);
    }
    void rotateDir(float phi){
        vec3 newDir = hypNormalize(hypRotate(pos, dir, phi));
        updatePosition(pos, newDir);
    }
    void updateGPU(){
        trail.updateGPU();
        baseCircle.updateGPU();
        leftEyeBgCircle.updateGPU();
        leftEyeCircle.updateGPU();
        rightEyeBgCircle.updateGPU();
        rightEyeCircle.updateGPU();
        mouthCircle.updateGPU();
    }
    void draw(){
        baseCircle.draw();
        leftEyeBgCircle.draw();
        leftEyeCircle.draw();
        rightEyeBgCircle.draw();
        rightEyeCircle.draw();
        mouthCircle.draw();
    }
};

struct RedHami : public HamiBase {
    RedHami()
    : HamiBase(
        vec3(0, 0, 1),
        vec3(0, 1, 0),
        vec3(1.0f, 0, 0)
    ) {}
};

struct GreenHami : public HamiBase {
    const vec2 start = vec2(1.0f, 1.0f);
    const vec2 ref = vec2(2.0f, 1.0f);
    GreenHami()
    : HamiBase(
      vec3(0, 0, 1),
      vec3(1, 0, 0),
      vec3(0, 1.0f, 0),
      false
    ) {
        // move to starting pos without trail, then enable trail
        rotateDir(-1.0f/(2.0f));
        moveForward(1.0f);
//        rotateDir(7.0f/(2.0f*M_PI));
        trailEnabled = true;
        moveForward(.5f);
    }
};

struct HamiGame {
    HypCircle backgroundDisk;
    RedHami redHami = RedHami();
    GreenHami greenHami = GreenHami();

    HamiGame()
    : backgroundDisk(
        HypCircle(
            vec3(0, 0, 1),
            vec3(1, 0, 0),
            6.0f,
            vec3(0, 0, 0)
        )
    ){
        redHami.otherEyePos = &greenHami.mouthCircle.pos;
        greenHami.otherEyePos = &redHami.mouthCircle.pos;

        backgroundDisk.updateGPU();
        redHami.updateGPU();
        greenHami.updateGPU();
    }

    void pulsateMouth(float elapsedTimeSec){
        redHami.pulsateMouth(elapsedTimeSec);
        greenHami.pulsateMouth(elapsedTimeSec);
    }

    void Draw(){
        redHami.calcEye();
        redHami.updateGPU();
        greenHami.calcEye();
        greenHami.updateGPU();

        backgroundDisk.draw();
        greenHami.trail.draw();
        redHami.trail.draw();
        greenHami.draw();
        redHami.draw();
    }
};


HamiGame* game;
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);

    game = new HamiGame();

	// create program for the GPU
	gpuProgram.create(vertexSource, fragmentSource, "outColor");

}

// Window has become invalid: Redraw
void onDisplay() {
	glClearColor(.5f, .5f, .5f, 1.0f);     // background color
	glClear(GL_COLOR_BUFFER_BIT); // clear frame buffer

    game->Draw();

	glutSwapBuffers(); // exchange buffers for double buffering
}

bool pressed[256] = { };
// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	//if (key == 'd') glutPostRedisplay();         // if d, invalidate display, i.e. redraw
    pressed[key] = true;
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {
    pressed[key] = false;
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {	// pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
	// Convert to normalized device space
	float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
	float cY = 1.0f - 2.0f * pY / windowHeight;
	printf("Mouse moved to (%3.2f, %3.2f)\n", cX, cY);
}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) { // pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
	// Convert to normalized device space
	float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
	float cY = 1.0f - 2.0f * pY / windowHeight;

    //char * buttonStat;
    std::string buttonStat;
	switch (state) {
	case GLUT_DOWN: buttonStat = "pressed"; break;
	case GLUT_UP:   buttonStat = "released"; break;
	}

	switch (button) {
	case GLUT_LEFT_BUTTON:   printf("Left button %s at (%3.2f, %3.2f)\n", buttonStat.c_str(), cX, cY);   break;
	case GLUT_MIDDLE_BUTTON: printf("Middle button %s at (%3.2f, %3.2f)\n", buttonStat.c_str(), cX, cY); break;
	case GLUT_RIGHT_BUTTON:  printf("Right button %s at (%3.2f, %3.2f)\n", buttonStat.c_str(), cX, cY);  break;
	}
}

long prevTime = 0;
const double tickRate = 120.0f;
const double tickTime = 1.0f/tickRate;
double tickCounter = 0, tickGoal = 0;
// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	long totalElapsedTime = glutGet(GLUT_ELAPSED_TIME); // elapsed time since the start of the program
    double totalElapsedTimeSec = totalElapsedTime / 1000.0;

    long deltaTime = totalElapsedTime - prevTime;
    prevTime = totalElapsedTime;
    double deltaTimeSec = deltaTime / 1000.0;

    tickGoal += ((double)deltaTimeSec / (double)tickTime);
    for (; tickCounter<tickGoal; tickCounter += 1.0){
        float tickDelta = tickTime*3.0f;

        game->greenHami.moveForward(tickDelta);
        game->greenHami.rotateDir(1.0f * tickDelta * (M_PI * 180.0f / 360.0f));

        if (pressed['f']){
            game->redHami.rotateDir(-1.0f * tickDelta * (M_PI * 180.0f / 360.0f));
        }
        if (pressed['s']){
            game->redHami.rotateDir(1.0f * tickDelta * (M_PI * 180.0f / 360.0f));
        }
        if (pressed['e']){
            game->redHami.moveForward(tickDelta);
        }
    }

    game->pulsateMouth(totalElapsedTimeSec);

    if (pressed['q']){
        game->redHami.trail.reset();
        game->greenHami.trail.reset();
    }

    glutPostRedisplay();
}
