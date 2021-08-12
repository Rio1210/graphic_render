//------------------------------------------------
//
//  img_paint
//
//
//-------------------------------------------------




#include <cmath>
#include <omp.h>
#include "imgproc.h"
#include "CmdLineFind.h"
#include <vector>
#include "fftanalysis.h"
#include "fftw3.h"

#include <GL/gl.h>   // OpenGL itself.
#include <GL/glu.h>  // GLU support library.
#include <GL/glut.h> // GLUT support library.


#include <iostream>
#include <stack>


using namespace std;
using namespace img;

ImgProc image;
ImgProc input;
int frame = 95;
vector<float> min_c;
vector<float> max_c;
vector<float> mean_c;
vector<float> standard_dev_c;
vector<vector<int>> histg1;

vector<vector<float>> CDF1;
vector<vector<int>> histg2;

vector<vector<float>> CDF2;
vector<vector<float>> PDF1;
vector<vector<float>> PDF2;
//LinearWave li(image, 3.33);
void img::unsmoothed_psd(const ImgProc &input, ImgProc &psd);

    void setNbCores(int nb)
{
   omp_set_num_threads( nb );
}

void cbMotion( int x, int y )
{
}

void cbMouse( int button, int state, int x, int y )
{
}

void cbDisplay( void )
{
   glClear(GL_COLOR_BUFFER_BIT );
   glDrawPixels( image.nx(), image.ny(), GL_RGB, GL_FLOAT, image.raw() );
   glutSwapBuffers();
}

void cbIdle()
{
   glutPostRedisplay();	
}

void cbOnKeyboard( unsigned char key, int x, int y )
{
   switch (key) 
   {

   case 'a':
   {
      cout << "  Linear Wave Estimation\n";
      ImgProc out;
      LinearWave li(image, 3.33);

      li.ingest(image);

      frame = 99;
      cout << "frame:  " << frame << endl;
      extrac_image(li, frame, image);
      image.flap();
   }
   break;
   case 'c':
      image.compliment();
      break;

   case 'V':
      //brit *=1.1;
      image.brightness(1.05);
      cout << "brightness increase\n";
      break;
   case 'v':
      //  brit *= 0.90;
      image.brightness(0.95);
      cout << "brightness decrease\n";
      break;
   case 'B':

      image.bias(0.05);
      cout << "bias increase\n";
      break;
   case 'b':

      image.bias(-0.05);
      cout << "bias decrease\n";
      break;

   case 'f':
      image.flap();
      cout << "flap\n";
      break;

   case 'G':
      // gm*=1.1;
      image.gamma(1.8);

      cout << "increase gamma\n";
      break;
   case 'g':

      image.gamma(0.95);
      cout << "decrease gamma\n";
      break;

   case 'h':
      image.histogram(histg1, CDF1, PDF1);
      histg1.clear();
      PDF1.clear();

      CDF1.clear();
      cout << "histogram\n";

      break;
   case 's':

      cout << "stat \n";

      image.stat();
      break;

   case 'z':
   {
      cout << "unsmooth\n";
      unsmoothed_psd(image, input);
      cout << "unsmooth\n";
      swap(image, input);
      }
         break;



      case 'r':
         image.org_data();
         cout << "original data\n"
              << endl;
         break;
   }
}

void PrintUsage()
{
   cout << "img_paint keyboard choices\n";
   cout << "c         compliment\n";
   cout << "a         Linear Wave Estimation\n";
   cout << "z         unsmooth\n";
   cout << "V         brightness increase\n";
   cout << "v         brightness decrease\n";
   cout << "B         bias increase\n";
   cout << "b         bias decrease\n";
   cout << "f         flap\n";
   cout << "G         increase gamma\n";
   cout << "g         decrease gamma\n";
   cout << "q         quantize\n";
   cout << "w         grayscale\n";
   cout << "R         original data\n";
   cout << "o         write out_file \n"
        << endl;
   cout << "h         histogram\n";
}



int main(int argc, char** argv)
{
   lux::CmdLineFind clf( argc, argv );

   setNbCores(8);

   string imagename = clf.find("-image", "", "Image to drive color");
   string whatsthis = clf.find("-stuff", "", "Image to drive color");

   cout << whatsthis << endl;

   clf.usage("-h");
   clf.printFinds();
   PrintUsage();

   image.load(imagename);

 
   // GLUT routines
   glutInit(&argc, argv);

   glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
   glutInitWindowSize( image.nx(), image.ny() );

   // Open a window 
   char title[] = "img_paint";
   glutCreateWindow( title );
   
   glClearColor( 1,1,1,1 );

   glutDisplayFunc(&cbDisplay);
   glutIdleFunc(&cbIdle);
   glutKeyboardFunc(&cbOnKeyboard);
   glutMouseFunc( &cbMouse );
   glutMotionFunc( &cbMotion );

   glutMainLoop();
   return 1;
};
