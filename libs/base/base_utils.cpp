
/*---------------------------------------------------------------------------*
 |                               UPGM++                                      |
 |                   Undirected Graphical Models in C++                      |
 |                                                                           |
 |              Copyright (C) 2014 Jose Raul Ruiz Sarmiento                  |
 |                 University of Malaga (jotaraul@uma.es)                    |
 |                                                                           |
 |   This program is free software: you can redistribute it and/or modify    |
 |   it under the terms of the GNU General Public License as published by    |
 |   the Free Software Foundation, either version 3 of the License, or       |
 |   (at your option) any later version.                                     |
 |                                                                           |
 |   This program is distributed in the hope that it will be useful,         |
 |   but WITHOUT ANY WARRANTY; without even the implied warranty of          |
 |   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the           |
 |   GNU General Public License for more details.                            |
 |   <http://www.gnu.org/licenses/>                                          |
 |                                                                           |
 *---------------------------------------------------------------------------*/

#include "base_utils.hpp"

/*Eigen::VectorXd UPGMpp::floatVectorToEigenDoubleVector( T &array )
{
    Eigen::VectorXd eigenVector;

    size_t N_elements = array.size();

    eigenVector.resize( N_elements );

    for ( size_t i = 0; i < N_elements; i++ )
        eigenVector( i ) = array[i];

    return eigenVector;
}*/

Eigen::VectorXd UPGMpp::logWithLove(Eigen::VectorXd &v)
{
    Eigen::VectorXd log;
    size_t N = v.rows();
    log.resize(N);

    for ( size_t i = 0; i < N; i++)
    {
        if ( !v(i) )
            log(i) = std::log(std::numeric_limits<double>::min());
        else
            log(i) = std::log(v(i));
    }

    return log;
}

//void fastLog(double x, double &logR)
//{
//    logR = 0;
//    for ( double n = 0; n < 1; n++ )
//        logR += (1/(2*n+1))*pow(((pow(x,2)-1)/(pow(x,2)+1)),2*n+1);
//}

void fastLog(double x, double &logR)
{
    for ( int n = 1; n < 15; n++ )
    {
        //cout << " " << logR << " " << (1/(2*n+1))*pow(((pow(x,2)-1)/(pow(x,2)+1)),2*n+1) << endl;
        //        logR += (1/(2*n+1))*pow(((pow(x,2)-1)/(pow(x,2)+1)),2*n+1);
        if ( n%2 )
            logR += (1/static_cast<double>(n))*pow(x-1,n);
        else
            logR -= (1/static_cast<double>(n))*pow(x-1,n);
    }

}

Eigen::MatrixXd UPGMpp::logWithLove(Eigen::MatrixXd &m)
{
    Eigen::MatrixXd log;
    size_t N_rows = m.rows();
    size_t N_cols = m.cols();

    log.resize(N_rows, N_cols);

    for ( size_t row = 0; row < N_rows; row++)
        for ( size_t col = 0; col < N_cols; col++ )
        {
            if ( !m(row,col) )
            {
                //                double logV;
                //                fastLog(1e-10,logV);
                //                log(row,col) = logV;

                log(row,col) = std::log(std::numeric_limits<double>::min());
            }
            else
            {
                //                double logV;
                //                fastLog(m(row,col),logV);
                //                log(row,col) = logV;

                log(row,col) = std::log(m(row,col));
            }
        }

    return log;
}

void UPGMpp::logWithLove(Eigen::MatrixXd &m, Eigen::MatrixXd &log)
{
    size_t N_rows = m.rows();
    size_t N_cols = m.cols();

    log.resize(N_rows, N_cols);

    for ( size_t row = 0; row < N_rows; row++)
        for ( size_t col = 0; col < N_cols; col++ )
        {
            if ( !m(row,col) )
            {
                                double logV;
                                fastLog(1e-10,logV);
                                log(row,col) = logV;

                //log(row,col) = std::log(std::numeric_limits<double>::min());
            }
            else
            {
                                double logV;
                                fastLog(m(row,col),logV);
                                log(row,col) = logV;

                //log(row,col) = std::log(m(row,col));
            }
        }

}
