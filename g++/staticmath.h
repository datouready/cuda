#pragma once

class staticmath
{
public:
    staticmath();
    ~staticmath();

    static double add(double a,double b);
    static double sub(double a,double b);
    static double mul(double a,double b);
    static double div(double a,double b);

    void print();
};