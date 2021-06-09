/**
* This file is part of ORB-SLAM.
* Reference: <https://github.com/bertabescos/DynaSLAM>. file MaskNet.h
* Copyright (C) 2021 Juyeb Shin <juyebshin@kaist.ac.kr> (Korean Advanced Institute of Science and Technology)
*
*/

#include "SemSeg.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <dirent.h>
#include <errno.h>

namespace ORB_SLAM2
{
#define U_SEGSt(a)\
    gettimeofday(&tvsv,0);\
    a = tvsv.tv_sec + tvsv.tv_usec/1000000.0
struct timeval tvsv;
double t1sv, t2sv,t0sv,t3sv;
void tic_initsv(){U_SEGSt(t0sv);}
void toc_finalsv(double &time){U_SEGSt(t3sv); time =  (t3sv- t0sv)/1;}
void ticsv(){U_SEGSt(t1sv);}
void tocsv(){U_SEGSt(t2sv);}

void SemSeg::ImportSettings()
{
    std::string strSettingsFile = "./Examples/SegSettings.yaml";
    cv::FileStorage fs(strSettingsFile.c_str(), cv::FileStorage::READ);
    fs["py_path"] >> this->py_path;
    fs["module_name"] >> this->module_name;
    fs["class_name"] >> this->class_name;
    fs["get_dyn_seg"] >> this->get_dyn_seg;

    std::cout << "    py_path: "<< this->py_path << std::endl;
    std::cout << "    module_name: "<< this->module_name << std::endl;
    std::cout << "    class_name: "<< this->class_name << std::endl;
    std::cout << "    get_dyn_seg: "<< this->get_dyn_seg << std::endl;
}

SemSeg::SemSeg()
{
    std::cout << "Importing Semantic Segementation Settings..." << std::endl;
    ImportSettings();
    std::cout << "Imported" << std::endl;
    std::string x;
    setenv("PYTHONPATH", this->py_path.c_str(), 1);
    x = getenv("PYTHONPATH");
    std::cout << "Environment: " << x << std::endl;
    try
    {
        /* code */
        Py_Initialize();
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
        PyErr_Print();
    }
    
    // PyObject *sysmodule = PyImport_ImportModule("sys");
    // PyObject *syspath = PyObject_GetAttrString(sysmodule, "path");
    // PyList_Append(syspath, PyUnicode_FromString("."));
    // Py_DECREF(syspath);
    // Py_DECREF(sysmodule);
    std::cout << "Python initialized" << std::endl;
    std::cout << Py_IsInitialized() << std::endl;
    PyRun_SimpleString("print(\'Hello Python\')\n");
    this->cvt = new NDArrayConverter();
    std::cout << "Converter allocated" << std::endl;
    // PyObject *pModule = PyUnicode_FromString(this->module_name.c_str());
    this->py_module = PyImport_ImportModule(this->module_name.c_str());
    std::cout << "Module check" << std::endl;
    std::cout << py_module << std::endl;
    if(py_module == 0)
    {
        std::cout << "ERROR importing module" << std::endl;
        PyErr_Print();
        exit(-1);
    }
    assert(py_module != 0);
    std::cout << "Module imported" << std::endl;
    this->py_class = PyObject_GetAttrString(this->py_module, this->class_name.c_str());
    assert(this->py_class != 0);
    std::cout << "Class imported" << std::endl;
    this->net = PyInstanceMethod_New(this->py_class);
    assert(this->net != 0);
    std::cout << "Instance Imported" << std::endl;
    std::cout << "Creating net instance..." << std::endl;
    std::string image  = "./Examples/000000.png";
    std::cout << "Loading net parameters..." << std::endl;
    GetSegmentation(image);
}

SemSeg::~SemSeg()
{
    delete this->py_module;
    delete this->py_class;
    delete this->net;
    delete this->cvt;
}

cv::Mat SemSeg::GetSegmentation(std::string &image, std::string dir, std::string rgb_name)
{
    // std::cout << "reading image: " << dir + "/" + name << std::endl;
    cv::Mat seg = cv::imread(image,CV_LOAD_IMAGE_UNCHANGED);
    // std::cout << "seg mat: " << seg.empty() << std::endl;
    if(seg.empty()){
        // PyObject* py_image = cvt->toNDArray(image.clone());
        // if(py_image == 0)
        // {
        //     std::cout << "ERROR converting toNDArray" << std::endl;
        //     PyErr_Print();
        //     exit(-1);
        // }
        // // std::cout << "toNDArray done" << std::endl;
        // assert(py_image != NULL);
        PyObject* py_string = PyUnicode_FromFormat(image.c_str());
        PyObject* py_mask_image = PyObject_CallMethod(this->net, const_cast<char*>(this->get_dyn_seg.c_str()),"(O)", py_string);
        // std::cout << "calling Method checking" << std::endl;
        if(py_mask_image == 0)
        {
            std::cout << "ERROR calling Method: " << get_dyn_seg << std::endl;
            PyErr_Print();
            exit(-1);
        }
        // std::cout << "calling Method done" << std::endl;
        seg = cvt->toMat(py_mask_image).clone();
        // std::cout << "seg mat: " << !seg.empty() << std::endl;
        seg.cv::Mat::convertTo(seg,CV_8UC3);// BGR
        if(dir.compare("no_save")!=0){
            DIR* _dir = opendir(dir.c_str());
            if (_dir) {closedir(_dir);}
            else if (ENOENT == errno)
            {
                const int check = mkdir(dir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
                if (check == -1) {
                    std::string str = dir;
                    str.replace(str.end() - 6, str.end(), "");
                    mkdir(str.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
                }
            }
            cv::imwrite(dir+"/"+rgb_name,seg);
        }
    }
    return seg;
}

} // namespace ORB_SLAM2