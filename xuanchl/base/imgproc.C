
#include <cmath>
#include <iostream>
#include <vector>
#include "imgproc.h"

#include <OpenImageIO/imageio.h>
OIIO_NAMESPACE_USING

using namespace img;
using namespace std;



ImgProc::ImgProc() :
  Nx (0),
  Ny (0),
  Nc (0),
  Nsize (0),
  img_data (nullptr)
{}

ImgProc::~ImgProc()
{
   clear();
}

void ImgProc::clear()
{
   if( img_data != nullptr ){ delete[] img_data; img_data = nullptr;}
   Nx = 0;
   Ny = 0;
   Nc = 0;
   Nsize = 0;
}

void ImgProc::clear(int nX, int nY, int nC)
{
   clear();
   Nx = nX;
   Ny = nY;
   Nc = nC;
   Nsize = (long)Nx * (long)Ny * (long)Nc;
   img_data = new float[Nsize];
#pragma omp parallel for
   for(long i=0;i<Nsize;i++){ img_data[i] = 0.0; }
}

bool ImgProc::load( const std::string& filename )
{
   auto in = ImageInput::create (filename);
   if (!in) {return false;}
   ImageSpec spec;
   in->open (filename, spec);
   clear();
   Nx = spec.width;
   Ny = spec.height;
   Nc = spec.nchannels;
   Nsize = (long)Nx * (long)Ny * (long)Nc;
   img_data = new float[Nsize];
   org_img_data = new float[Nsize];
   in->read_image(TypeDesc::FLOAT, img_data);
   in->read_image(TypeDesc::FLOAT, org_img_data);
   in->close ();
   cout << "Nx:  " << Nx << "  Ny:  "<< Ny  <<"   Nc: "<< Nc<< endl;


   return true;
}

void ImgProc::write(){
  const char* file_name = "output.exr";
  auto out = ImageOutput::create(file_name);
  if(!out){
    return;
  }
    ImageSpec spe(Nx, Ny, Nc, TypeDesc::FLOAT);
    out->open (file_name, spe);
    out->write_image(TypeDesc::FLOAT, img_data);
    out->close();
  }

//one pixel value, include RGB...
void ImgProc::value( int i, int j, std::vector<float>& pixel) const
{

   pixel.clear();
   if( img_data == nullptr ){ return; }
   if( i<0 || i>=Nx ){ return; }
   if( j<0 || j>=Ny ){ return; }
   pixel.resize(Nc);
   for( int c=0;c<Nc;c++ )
   {
      pixel[c] = img_data[index(i,j,c)];
   }
   return;
}
//set one value include RGB...
void ImgProc::set_value( int i, int j, const std::vector<float>& pixel)
{

   if( img_data == nullptr ){ return; }
   if( i<0 || i>=Nx ){ return; }
   if( j<0 || j>=Ny ){ return; }
   if( Nc > (int)pixel.size() ){ return; }
#pragma omp parallel for
   for( int c=0;c<Nc;c++ )
   {
      img_data[index(i,j,c)] = pixel[c];
   }
   return;
}


ImgProc::ImgProc(const ImgProc& v) :
  Nx (v.Nx),
  Ny (v.Ny),
  Nc (v.Nc),
  Nsize (v.Nsize)
{
   img_data = new float[Nsize];
#pragma omp parallel for
   for( long i=0;i<Nsize;i++){ img_data[i] = v.img_data[i]; }
}

ImgProc& ImgProc::operator=(const ImgProc& v)
{
   if( this == &v ){ return *this; }
   if( Nx != v.Nx || Ny != v.Ny || Nc != v.Nc )
   {
      clear();
      Nx = v.Nx;
      Ny = v.Ny;
      Nc = v.Nc;
      Nsize = v.Nsize;
   }
   img_data = new float[Nsize];
#pragma omp parallel for
   for( long i=0;i<Nsize;i++){ img_data[i] = v.img_data[i]; }
   return *this;
}


void ImgProc::operator*=(float v)
{
   if( img_data == nullptr ){ return; }
#pragma omp parallel for
   for( long i=0;i<Nsize;i++ ){ img_data[i] *= v; }
}

void ImgProc::operator/=(float v)
{
   if( img_data == nullptr ){ return; }
#pragma omp parallel for
   for( long i=0;i<Nsize;i++ ){ img_data[i] /= v; }
}

void ImgProc::operator+=(float v)
{
   if( img_data == nullptr ){ return; }
#pragma omp parallel for
   for( long i=0;i<Nsize;i++ ){ img_data[i] += v; }
}

void ImgProc::operator-=(float v)
{
   if( img_data == nullptr ){ return; }
#pragma omp parallel for
   for( long i=0;i<Nsize;i++ ){ img_data[i] -= v; }
}

/*******************************************************/
/*******************************************************/

void ImgProc::compliment()
{
   if( img_data == nullptr ){ return; }
#pragma omp parallel for
   for( long i=0;i<Nsize;i++ ){ img_data[i] = 1.0 - img_data[i]; }
}


void ImgProc::brightness(float brit) {
  if( img_data == nullptr ){ return; }
  #pragma omp parallel for
  for( long i=0;i<Nsize;i++ ){
    
    img_data[i] = img_data[i]*brit;
    if(img_data[i]<0.0)
	img_data[i] = 0.0;
    else if(img_data[i]>1.0)
	img_data[i] = 1.0;
	
    }
}


void ImgProc::bias(float bis) {
  #pragma omp parallel for
    for( long i=0;i<Nsize;i++ ){
      img_data[i] = img_data[i] + bis;

      if(img_data[i]<0.0)
	img_data[i] = 0.0;
      else if(img_data[i]>1.0)
	img_data[i] = 1.0;
    }
}

void ImgProc::gamma (float gm) {
	
  #pragma omp parallel for
    for( long i=0;i<Nsize;i++ ){
      img_data[i] = pow(img_data[i],gm);
    }
}



void ImgProc::flap() {
    #pragma omp parallel for
  for(int j=0;j<Ny/2;j++)
  {
  #pragma omp parallel for
   for(int i=0;i<Nx;i++)
   {
      vector<float> C;
      vector<float> C2;
      value(i,j,C);
      value(i,Ny-j-1,C2);
      set_value(i,Ny-j-1,C);
      set_value(i,j,C2);
   }
  }

}

void ImgProc::grayscale() {


   //only need 2 pragma //always 3 channels
   float g = 0.0;
     #pragma omp parallel for
     for(long j = 0; j < Ny; j++ ) {
        #pragma omp parallel for
        for(long i = 0; i < Nx; i++) {
        vector<float> p_c(Nc);
        value(i,j,p_c);
        g = p_c[0]*0.2126 + p_c[1]*0.7152 + p_c[2]*0.0722;

        for(unsigned int c = 0; c < p_c.size(); c++){
           p_c[c] = g;
         }
         set_value(i,j,p_c);
       }
     }
   }
   void ImgProc::org_data()
   {
     for (long i = 0; i < Nsize; i++)
     {
       img_data[i] = org_img_data[i];
     }
   }

//calculate each channel 0<data<1 every R, G, and B
//result = floor(data*at[i][j])/data*at[i][j]
void ImgProc::quantize () {

   #pragma omp parallel for
   for(long i = 0; i < Ny; i++ ) {
     #pragma omp parallel for
     for(long j = 0; j < Nx; j++) {
       vector<float> p_c(Nc);
       value(j,i,p_c);
     //  #pragma omp parallel for
       for(long c = 0; c < Nc; c++) {
         int tmp = p_c[c]*(i*Ny + j);
          p_c[c] = tmp/(float)(i*Ny + j);
       }
       set_value(j,i,p_c);
     }
   }
}



void ImgProc::rms() {
    //step 1   computing mean
    float sum_r = 0.0;
    float sum_g = 0.0;
    float sum_b = 0.0;
    #pragma omp parallel for
    for(long i = 0; i < Ny; i++) {
        #pragma omp parallel for
        for(long j = 0; j < Nx; j++) {
            vector<float> p_c(Nc);
            value (j, i, p_c);

                sum_r += p_c[0];
                sum_g += p_c[1];
                sum_b += p_c[2];
        }
    }
    float mean_r = sum_r/(Nx*Ny);
    float mean_g = sum_g/(Nx*Ny);
    float mean_b = sum_b/(Nx*Ny);
    //step2 computing sigma
    float vari_r = 0;
    float vari_g = 0;
    float vari_b = 0;
    #pragma omp parallel for
    for(long i = 0; i < Ny; i++) {
        #pragma omp parallel for
        for(long j = 0; j < Nx; j++) {
            vector<float> p_c(Nc);
            value (j, i, p_c);
            vari_r += pow((p_c[0] - mean_r),2);
            vari_g += pow((p_c[1] - mean_g),2);
            vari_b += pow((p_c[2] - mean_b),2);
        }
    }

    float sigma_r = sqrt(vari_r/(Nx*Ny));
    float sigma_g = sqrt(vari_g/(Nx*Ny));
    float sigma_b = sqrt(vari_b/(Nx*Ny));

    //step 3  computing contrast and set value to data
    #pragma omp parallel for
    for(long i = 0; i < Ny; i++) {
        #pragma omp parallel for
        for(long j = 0; j < Nx; j++) {
          vector<float> p_c(Nc);
          value(j,i,p_c);
          p_c[0] = (p_c[0] - mean_r)/sigma_r;
          p_c[1] = (p_c[1] - mean_g)/sigma_g;
          p_c[2] = (p_c[2] - mean_b)/sigma_b;
          set_value(j,i,p_c);

       }
    }
}



void ImgProc::stat(){

  
  vector<vector<float>> his_pix;
 
  //  #pragma omp parallel for
  for(long i = 0; i < Ny; i++) {
    for(long j = 0; j < Nx; j++) {
      vector<float> his_c(Nc);// = {4.0,5.0,6.0};
      value(j, i, his_c);

      his_pix.push_back(his_c);
    }
    cout<<endl;
  }
  vector<float> min;
  vector<float> max;
  cout << "   	                Red           Green          Blue \n";
  cout << "   min/max:";
    
 // #pragma omp parallel for
  for(long i = 0; i < Nc; i++) {
    float max_c= 0.0;
    float min_c = 1.0;
   
    for(long j = 0; j < Nx*Ny; j++){
      float h = his_pix[j][i];// * 1000.0;
      min_c = (min_c > h)? h : min_c;
      max_c =  (max_c < h)? h : max_c;
     
    }
    cout <<"             " << min_c << "/" << max_c;
    min.push_back(min_c);
    max.push_back(max_c);

  }
  cout << endl;
  //--------------------step2: get mean from each channels
  // vector: mean(R_mean, G_mean, B_mean )
  //vector<float> mean(Nc);
  float tol_I = 0.0;
  vector<float> mean;
  cout <<"   mean:        ";
  //#pragma omp parallel for
  for(long i = 0; i < Nc; i++) {
    //#pragma omp parallel for
    for(long j = 0; j < Nx*Ny; j++) {
      tol_I += his_pix[j][i];
    }
   float mean_c = tol_I / (Nx*Ny);
   cout << "       " << mean_c; 
   mean.push_back(mean_c);
 
  }
   cout << endl;

  //-----------------------step3: get standart dev
  //vector: dev(R_dev,G_dev,B_dev)
  vector<float> standard_dev;
  float var_I = 0.0;
  float dev_c = 0.0;
 cout<< "   stand_devation:";

  //get dev form each channel
  //vector<float> dev(Nc);

  for(long i = 0; i < Nc; i++ ) {
  //  #pragma omp parallel for
    for(long j = 0; j < Nx*Ny; j++) {
      //(x-mean_x)^2 total
      var_I += pow((his_pix[j][i] - mean[i]),2);

    }
    dev_c = sqrt(var_I/(Nx*Ny));
    cout << "    " << dev_c <<"   ";
    standard_dev.push_back(dev_c);

  }
   cout << endl;
}


void ImgProc::histogram(vector<vector<int>>& histg, vector<vector<float>>& CDF, vector<vector<float>>& PDF){


  vector<vector<float>> his_pix;

  //cypr orginal pixel to a new vector for use(compute)
  // #pragma omp parallel for
  for(long i = 0; i < Ny; i++) {
   //   #pragma omp parallel for
    for(long j = 0; j < Nx; j++) {
      vector<float> his_c(Nc);// = {4.0,5.0,6.0};
      value(j, i, his_c);

      his_pix.push_back(his_c);
 
    }
  }

  //-------------------- step1: get R G B A ... max min from each channel
  vector<float> min; // RGBA...
  vector<float> max; // RGBA..
 // #pragma omp parallel for
  for(long i = 0; i < Nc; i++) {
    float max_c= 0.0;
    float min_c = 1.0;
   // #pragma omp parallel for
    for(long j = 0; j < Nx*Ny; j++){
      float h = his_pix[j][i];// * 1000.0;
      min_c = (min_c > h)? h : min_c;
      max_c =  (max_c < h)? h : max_c;
     
    }

   
    min.push_back(min_c);
    max.push_back(max_c);

  }


      //N = number
      //save delta I from each channel RGBA...
    vector<float> delta_Inten;// = (max[i] - min[i]) / 256.0;
    //#N = 256
    for(long i = 0; i < Nc; i++) {
      delta_Inten.push_back((max[i] - min[i])/256.0);// / 256.0);
    }

  vector< int > histg_c(256, 0);
 // vector< vector<int> > histg;
  for(long i = 0; i < Nc; i++) {
    for(long j = 0; j < Nx*Ny; j++) {

      int m = (his_pix[j][i] - min[i])/delta_Inten[i];

      histg_c[m]++;
    }
    //how many pixel have intensity in range m  RGBA each channel

    histg.push_back(histg_c);
    fill(histg_c.begin(), histg_c.end(),0);
  }

    vector<float> PDF_c(256,0.0);
	//vector PDF: (R , G , B)
   // vector< vector<float> > PDF;
    
    for(long i = 0; i < Nc; i++) {
      float count = 0.0;
      for(long j = 0; j < 256; j++) {

        PDF_c[j] = histg[i][j]/(float)(Nx*Ny);
 
     count = count + PDF_c[j];

      }
      
      PDF.push_back(PDF_c);
      fill(PDF_c.begin(), PDF_c.end(), 0);
}
    vector<float> CDF_c(256,0.0);
    //vector< vector<float> > CDF;

    for(long i = 0; i < Nc; i++) {
      CDF_c[0] = PDF[i][0];

      for(long j = 1; j < 256; j++) {
        CDF_c[j] = CDF_c[j-1] + PDF[i][j];
    }

      CDF.push_back(CDF_c);
      fill(CDF_c.begin(), CDF_c.end(), 0);

    }


  vector<float> eq_c(Nx*Ny);
  vector< vector<float> > eq;

 //#pragma omp parallel for
  for(long i = 0; i < Nc; i++) {
    //  #pragma omp parallel for
    for(long j = 0; j < Nx*Ny; j++) {
      float Q = (his_pix[j][i] - min[i])/delta_Inten[i];
      int q = int(Q);
	//cout << q << "   q  " ;


      float w = Q-q;
      float eq_I = 0.0;
      if(q < 255) {
        eq_I = CDF[i][q]*(1 - w) + CDF[i][q+1] * w;
      } else if(q == 255) {
        eq_I = CDF[i][q];
      }

      eq_c[j] = eq_I;
    }

    eq.push_back(eq_c);
    fill(eq_c.begin(), eq_c.end(), 0);

}

//=================================

 // #pragma omp parallel for
  for(long i = 0; i < Ny; i++) {
    for(long j = 0; j < Nx; j++) {
      vector<float> p_c;//(Nc);
      for(long k = 0; k < Nc; k++) {
        p_c.push_back(eq[k][i*Nx + j]);
      }
      set_value(j,i,p_c);
    }
  }
}


/**************************************************/
/*********************************************/

long ImgProc::index(int i, int j, int c) const
{
   return (long) c + (long) Nc * index(i,j); // interleaved channels

   // return index(i,j) + (long)Nx * (long)Ny * (long)c; // sequential channels
}

long ImgProc::index(int i, int j) const
{
   return (long) i + (long)Nx * (long)j;
}

void img::swap(ImgProc& u, ImgProc& v)
{
   float* temp = v.img_data;
   int Nx = v.Nx;
   int Ny = v.Ny;
   int Nc = v.Nc;
   long Nsize = v.Nsize;

   v.Nx = u.Nx;
   v.Ny = u.Ny;
   v.Nc = u.Nc;
   v.Nsize = u.Nsize;
   v.img_data = u.img_data;

   u.Nx = Nx;
   u.Ny = Ny;
   u.Nc = Nc;
   u.Nsize = Nsize;
   u.img_data = temp;
}
