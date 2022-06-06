#include "dynamicmath.h"
#include <iostream>
using namespace std;

int main(int argc,char* argv[])
{
    double a=10;
    double b=2;
    cout<<"a+b="<<dynamicmath::add(a,b)<<endl;
    return 0;  
}