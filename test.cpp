#include "llsvm.h"

#include <cstdlib>
#include <iostream>
#include <fstream>
using namespace std;

double frand(){ return (double)rand()/RAND_MAX; }

int main()
{
  libsvm::svm s;
  s.set_nu_svc();
  
  for (int i=0; i<1000; i++){
    double x=frand()*2-1;
    double y=frand()*2-1;
    vector<pair<int, double> > v;
    v.push_back(make_pair(0, x));
    v.push_back(make_pair(1, y));
    s.add_train_data(x*y>=0, v);
  }
  
  s.train();

  for (int i=0; i<10; i++){
    double x=frand()*2-1;
    double y=frand()*2-1;
    vector<pair<int, double> > v;
    v.push_back(make_pair(0, x));
    v.push_back(make_pair(1, y));
    cout<<x<<", "<<y<<": "<<s.predict(v)<<endl;
  }
  return 0;
}
