#pragma once

#include <libsvm/svm.h>

#include <vector>
#include <set>
#include <string>
#include <iostream>
#include <cstdio>
#include <malloc.h>

namespace libsvm{

class svm{
public:
  svm()
    :model(NULL){

    param.svm_type=C_SVC;
    param.kernel_type=RBF;
    param.degree=3;
    param.gamma=-1; // 1/num_features
    param.coef0=0;
    param.cache_size=100;
    param.eps=0.001;
    param.C=1;
    param.nr_weight=0;
    param.weight_label=NULL;
    param.weight=NULL;
    param.nu=0.5;
    param.p=0.1;
    param.shrinking=1;
    param.probability=0;
  }

  explicit svm(const std::string &fname){
    model=svm_load_model(fname.c_str());
  }

  ~svm(){
    if (model){
      svm_destroy_model(model);
      model=NULL;
    }
  }

  void set_linear(){
    param.kernel_type=LINEAR;
  }

  void set_rbf(double gamma=-1){
    param.kernel_type=RBF;
    param.gamma=gamma;
  }

  void set_sigmoid(double gamma=-1, double coef0=0){
    param.kernel_type=SIGMOID;
    param.gamma=gamma;
    param.coef0=coef0;
  }

  void set_poly(double gamma=-1, double coef0=0, double degree=3){
    param.kernel_type=POLY;
    param.gamma=gamma;
    param.coef0=coef0;
    param.degree=degree;
  }

  void set_c_svc(double C=1){
    param.svm_type=C_SVC;
    param.C=C;
  }

  void set_nu_svc(double nu=0.5){
    param.svm_type=NU_SVC;
    param.nu=nu;
  }

  void set_nu_svr(double nu=0.5, double C=1){
    param.svm_type=NU_SVR;
    param.nu=nu;
    param.C=C;
  }

  void set_epsilon_svr(double eps=0.001, double C=1){
    param.svm_type=EPSILON_SVR;
    param.C=C;
  }

  void set_one_class(double nu=0.5){
    param.svm_type=ONE_CLASS;
    param.nu=nu;
  }

  void set_cache_size(double mb){
    param.cache_size=mb;
  }

  void set_shrinking(bool b){
    param.shrinking=b?1:0;
  }

  void set_probability(bool b){
    param.probability=b?1:0;
  }

  void save(const std::string &fname) const {
    svm_save_model(fname.c_str(), model);
  }

  void add_train_data(double ans, const std::vector<std::pair<int, double> > &vect){
    dat.push_back(make_pair(ans, vect));
  }

  void train(bool auto_reload=true){
    std::set<int> feats;
    for (size_t i=0; i<dat.size(); i++)
      for (size_t j=0; j<dat[i].second.size(); j++)
	feats.insert(dat[i].second[j].first);

    if (param.gamma==-1)
      param.gamma=1.0/(std::max(1ul, feats.size()));

    svm_problem prob;

    prob.l=dat.size();
    prob.y=new double[dat.size()];
    for (size_t i=0; i<dat.size(); i++)
      prob.y[i]=dat[i].first;
    prob.x=new svm_node*[dat.size()];
    for (size_t i=0; i<dat.size(); i++){
      prob.x[i]=new svm_node[dat[i].second.size()+1];
      for (size_t j=0; j<dat[i].second.size(); j++){
	prob.x[i][j].index=dat[i].second[j].first;
	prob.x[i][j].value=dat[i].second[j].second;
      }
      prob.x[i][dat[i].second.size()].index=-1;
      prob.x[i][dat[i].second.size()].value=0;
    }

    model=svm_train(&prob, &param);

    if (auto_reload){
      char *p=tempnam(NULL, "llsvmtmp");
      this->save(p);

      svm(p).swap(*this);
      
      unlink(p);
      free(p);
    }

    delete []prob.y;
    for (int i=0; i<prob.l; i++)
      delete []prob.x[i];
    delete []prob.x;
  }

  double predict(const std::vector<std::pair<int, double> > &vect) const {
    std::vector<svm_node> node(vect.size()+1);
    for (size_t i=0; i<vect.size(); i++){
      node[i].index=vect[i].first;
      node[i].value=vect[i].second;
    }
    node[vect.size()].index=-1;
    node[vect.size()].value=0;

    return svm_predict(model, &node[0]);
  }

  void swap(svm &r) throw() {
    dat.swap(r.dat);
    std::swap(param, r.param);
    std::swap(model, r.model);
  }

  std::vector<std::pair<int, std::vector<std::pair<int, double> > > > dat;
  svm_parameter param;
  svm_model *model;
};

} // libsvm
