/*
* MIT License (modified header for fixed-wing conversion)
*
* This file is adapted from crazyflie_nmpc to a simple fixed-wing NMPC node.
* - States: x,y,z, u,v,w, p,q,r, phi,theta,psi, gamma
* - Controls: delta_e, delta_a, delta_r, delta_t
* - Dynamics: small-angle placeholder (replace with CasADi/acados model)
*
* Important: generate acados model & solver for your fixed-wing dynamics and
* update included headers below (see TODO markers).
*/

#include <ros/ros.h>
#include <std_srvs/Empty.h>

// msgs
#include <std_msgs/String.h>
#include <std_msgs/Float32MultiArray.h>
#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/Twist.h>

// Use the same estimator message type you had (adapt if needed)
#include <crazyflie_controller/CrazyflieStateStamped.h>

// Dynamic reconf
#include <dynamic_reconfigure/server.h>
#include <boost/thread.hpp>
#include "boost/thread/mutex.hpp"
// keep the crazyflie_paramsConfig for dynamic params (or create new)
#include <crazyflie_controller/crazyflie_paramsConfig.h>

// Matrices and vectors
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>

// standard
#include <iostream>
#include <sstream>
#include <fstream>
#include <ios>

// acados
#include "acados/utils/print.h"
#include "acados_c/ocp_nlp_interface.h"
#include "acados_c/external_function_interface.h"
#include "acados/ocp_nlp/ocp_nlp_constraints_bgh.h"
#include "acados/ocp_nlp/ocp_nlp_cost_ls.h"

// blasfeo
#include "blasfeo/include/blasfeo_d_aux.h"
#include "blasfeo/include/blasfeo_d_aux_ext_dep.h"

// ----------------- TODO: Replace the two includes below with your generated fixed-wing model+solver
// e.g. include "fixedwing_model/fixedwing_model.h" and "acados_solver_fixedwing.h"
// For now these includes are placeholders and will need the generated files to compile and link.
#include "fixedwing_model/fixedwing_model.h"            // <-- generate with CasADi/acados
#include "acados_solver_fixedwing.h"                    // <-- generate with acados

// global data (acados)
ocp_nlp_in * nlp_in;
ocp_nlp_out * nlp_out;
ocp_nlp_solver * nlp_solver;
void * nlp_opts;
ocp_nlp_plan * nlp_solver_plan;
ocp_nlp_config * nlp_config;
ocp_nlp_dims * nlp_dims;

external_function_param_casadi * forw_vde_casadi;

using namespace Eigen;
using std::ofstream;
using std::cout;
using std::endl;
using std::fixed;
using std::showpos;

// acados dims and constants
#define N 50         // horizon length
#define NX 13        // number of states (12 + gamma)
#define NU 4         // delta_e, delta_a, delta_r, delta_t
#define NY (NX + NU) // stage measurement dimension
#define NYN (NX)     // terminal measurement dimension

#define pi  3.14159265358979323846
#define g0 9.80665

#define WEIGHT_MATRICES 0
#define SET_WEIGHTS 0
#define FIXED_U0 0
#define CONTROLLER 1
#define PUB_OPENLOOP_TRAJ 0

class NMPC
{
    // State enum (body-frame velocities)
    enum systemStates{
        x   = 0,
        y   = 1,
        z   = 2,
        u   = 3,  // body-forward velocity
        v   = 4,  // body-side velocity
        w   = 5,  // body-down velocity
        p   = 6,  // roll rate
        q   = 7,  // pitch rate
        r   = 8,  // yaw rate
        phi = 9,
        th  = 10, // theta (pitch)
        psi = 11, // yaw
        gamma = 12 // flight-path angle (gamma)
    };

    enum controlInputs{
        delta_e = 0,
        delta_a = 1,
        delta_r = 2,
        delta_t = 3
    };

    enum reference_mode{
        Regulation = 0,
        Tracking = 1,
        Position_Hold = 2
    };

    struct solver_output{
        double status, KKT_res, cpu_time;
        double u0[NU];
        double u1[NU];
        double x1[NX];
        double x2[NX];
        double x4[NX];
        double xi[NU];
        double ui[NU];
    };

    struct solver_input{
        double x0[NX];
        double yref[(NY*N)];
        double yref_e[NYN];
        double W[NY*NY];
        double WN[NX*NX];
    };

    ros::Publisher p_ctrl_surf;   // publishes control surfaces (delta_e, delta_a, delta_r, delta_t)
    ros::Publisher p_bodycmd;    // publishes a simplified control command for lower-level autopilot (deg/%)

    ros::Subscriber s_estimator;  // estimator subscriber (adapt to your estimator topic)

    // NMPC variables
    unsigned int k,i,j,ii;
    double Wdiag_x,Wdiag_y,Wdiag_z;
    double Wdiag_u,Wdiag_v,Wdiag_w;
    double Wdiag_p,Wdiag_q,Wdiag_r;
    double Wdiag_phi,Wdiag_th,Wdiag_psi;
    double Wdiag_gamma;
    double Wdiag_de,Wdiag_da,Wdiag_dr,Wdiag_dt;
    double WN_factor;

    // acados struct
    solver_input acados_in;
    solver_output acados_out;
    int acados_status;

    reference_mode policy;

    // trajectory storage (optional)
    std::vector<std::vector<double>> precomputed_traj;
    int N_STEPS, iter;

public:
    NMPC(ros::NodeHandle& n, const std::string& ref_traj)
    {
        int status = 0;
        WN_factor = 50.0;

        // Attempt to create acados objects (acados_create is from generated solver)
        status = acados_create();
        if (status){
            ROS_INFO_STREAM("acados_create() returned status " << status << ". Exiting." << endl);
            exit(1);
        }

        // publishers
        p_ctrl_surf = n.advertise<std_msgs::Float32MultiArray>("/fixedwing/control_surfaces", 1);
        p_bodycmd = n.advertise<geometry_msgs::Twist>("/fixedwing/body_cmd", 1);

        // subscriber of estimator state - adapt topic and message type to your estimator
        s_estimator = n.subscribe("/cf_estimator/state_estimate", 5, &NMPC::iteration, this);

        // initialize steady-state (trim) or defaults
        iter = 0;
        N_STEPS = 0;

        // set reasonable weight defaults - tune for your aircraft!
        Wdiag_x = 100.0;
        Wdiag_y = 100.0;
        Wdiag_z = 200.0;

        Wdiag_u = 1.0;
        Wdiag_v = 1.0;
        Wdiag_w = 5.0;

        Wdiag_p = 0.1;
        Wdiag_q = 0.1;
        Wdiag_r = 0.1;

        Wdiag_phi = 1.0;
        Wdiag_th  = 1.0;
        Wdiag_psi = 1.0;

        Wdiag_gamma = 50.0;

        Wdiag_de = 0.1;
        Wdiag_da = 0.1;
        Wdiag_dr = 0.1;
        Wdiag_dt = 0.1;

        // initialize acados in/out arrays to zero
        for (ii=0; ii<NX; ii++) acados_in.x0[ii] = 0.0;
        for (ii=0; ii<NY* N; ii++) acados_in.yref[ii] = 0.0;
        for (ii=0; ii<NYN; ii++) acados_in.yref_e[ii] = 0.0;
    }

    void run()
    {
        ROS_DEBUG("Setting up dynamic reconfigure and spinning");
        dynamic_reconfigure::Server<crazyflie_controller::crazyflie_paramsConfig> server;
        dynamic_reconfigure::Server<crazyflie_controller::crazyflie_paramsConfig>::CallbackType f;
        f = boost::bind(&NMPC::callback_dynamic_reconfigure, this, _1, _2);
        server.setCallback(f);

        ros::spin();
    }

    void callback_dynamic_reconfigure(crazyflie_controller::crazyflie_paramsConfig &config, uint32_t level)
    {
        if (level && CONTROLLER)
        {
            if(config.enable_traj_tracking)
            {
                ROS_INFO_STREAM("Tracking trajectory");
                config.enable_regulation = false;
                policy = Tracking;
            }
            if(config.enable_regulation)
            {
                config.enable_traj_tracking = false;
                policy = Regulation;
            }
            ROS_INFO_STREAM(fixed << showpos << "Fixed-wing NMPC status: "
                << (config.enable_regulation?"Regulation ":"")
                << (config.enable_traj_tracking?"Tracking ":"")
                << endl);
        }
    }

    int readDataFromFile(const char* fileName, std::vector<std::vector<double>> &data)
    {
        std::ifstream file(fileName);
        std::string line;
        int num_of_steps = 0;

        if (file.is_open())
        {
            while(getline(file, line)){
                ++num_of_steps;
                std::istringstream linestream( line );
                std::vector<double> linedata;
                double number;
                while( linestream >> number ){
                    linedata.push_back( number );
                }
                data.push_back( linedata );
            }
            file.close();
        }
        else
        {
            return 0;
        }
        return num_of_steps;
    }

    // convert quaternion -> euler (same as before)
    struct euler { double phi; double theta; double psi; };
    euler quatern2euler(Quaterniond* q)
    {
        euler angle;
        double R11 = 2*(q->w()*q->w()+q->x()*q->x())-1;
        double R21 = 2*(q->x()*q->y()-q->w()*q->z());
        double R31 = 2*(q->x()*q->z()+q->w()*q->y());
        double R32 = 2*(q->y()*q->z()-q->w()*q->x());
        double R33 = 2*(q->w()*q->w()+q->z()*q->z())-1;

        double phi = atan2(R32, R33);
        double theta = -asin(R31);
        double psi = atan2(R21, R11);

        angle.phi = phi;
        angle.theta = theta;
        angle.psi = psi;
        return angle;
    }

    double rad2Deg(double rad) { return rad * 180.0 / pi; }
    double deg2Rad(double deg) { return deg / 180.0 * pi; }

    void nmpcReset()
    {
        acados_free();
    }

    // small helper: gamma from (u,w) in body frame (note sign conventions may vary by your NED/ENU choice)
    double compute_gamma(double u_body, double w_body)
    {
        // flight-path angle gamma = atan2(-w, u) if w is body-down (positive down)
        // modify sign according to your velocity sign convention
        return atan2(-w_body, u_body);
    }

    // -------------------------------------------------------------
    // Main iteration callback: reads estimator message and runs NMPC
    // -------------------------------------------------------------
    void iteration(const crazyflie_controller::CrazyflieStateStampedPtr& msg)
    {
        try {
            // --- Build references according to policy
            if (policy == Regulation)
            {
                // simple regulation to desired (use dynamic_reconfigure to set desired in a real setup)
                for (k = 0; k < N+1; k++)
                {
                    // minimal example: hold current position as reference
                    acados_in.yref[k*NY + 0] = msg->pos.x; // x
                    acados_in.yref[k*NY + 1] = msg->pos.y; // y
                    acados_in.yref[k*NY + 2] = msg->pos.z; // z
                    // keep velocities zero
                    acados_in.yref[k*NY + 3] = 0.0; // u
                    acados_in.yref[k*NY + 4] = 0.0; // v
                    acados_in.yref[k*NY + 5] = 0.0; // w
                    // rates zero
                    acados_in.yref[k*NY + 6] = 0.0; // p
                    acados_in.yref[k*NY + 7] = 0.0; // q
                    acados_in.yref[k*NY + 8] = 0.0; // r
                    // attitudes keep current
                    // We'll fill attitudes at the state update section below
                    // controls: nominal zero
                    acados_in.yref[k*NY + 13] = 0.0; // delta_e
                    acados_in.yref[k*NY + 14] = 0.0; // delta_a
                    acados_in.yref[k*NY + 15] = 0.0; // delta_r
                    acados_in.yref[k*NY + 16] = 0.5; // delta_t (50% throttle nominal)
                }
            }
            else if (policy == Tracking)
            {
                // tracking logic if you provide precomputed_traj (similar to original code)
                // left as exercise / future expansion
            }
            else // Position_Hold
            {
                // similar to regulation but using last precomputed point
            }

            // --- Fill weight matrices (diagonal)
            for (ii = 0; ii < (NY*NY); ii++) acados_in.W[ii] = 0.0;
            for (ii = 0; ii < (NX*NX); ii++) acados_in.WN[ii] = 0.0;

            // populate W (NOTE: layout assumed row-major via index formula used in original)
            acados_in.W[0 + 0*(NU+NX)] = Wdiag_x;
            acados_in.W[1 + 1*(NU+NX)] = Wdiag_y;
            acados_in.W[2 + 2*(NU+NX)] = Wdiag_z;
            acados_in.W[3 + 3*(NU+NX)] = Wdiag_u;
            acados_in.W[4 + 4*(NU+NX)] = Wdiag_v;
            acados_in.W[5 + 5*(NU+NX)] = Wdiag_w;
            acados_in.W[6 + 6*(NU+NX)] = Wdiag_p;
            acados_in.W[7 + 7*(NU+NX)] = Wdiag_q;
            acados_in.W[8 + 8*(NU+NX)] = Wdiag_r;
            acados_in.W[9 + 9*(NU+NX)] = Wdiag_phi;
            acados_in.W[10 + 10*(NU+NX)] = Wdiag_th;
            acados_in.W[11 + 11*(NU+NX)] = Wdiag_psi;
            acados_in.W[12 + 12*(NU+NX)] = Wdiag_gamma;
            acados_in.W[13 + 13*(NU+NX)] = Wdiag_de;
            acados_in.W[14 + 14*(NU+NX)] = Wdiag_da;
            acados_in.W[15 + 15*(NU+NX)] = Wdiag_dr;
            acados_in.W[16 + 16*(NU+NX)] = Wdiag_dt;

            // terminal weights
            acados_in.WN[0 + 0*(NX)] = Wdiag_x * WN_factor;
            acados_in.WN[1 + 1*(NX)] = Wdiag_y * WN_factor;
            acados_in.WN[2 + 2*(NX)] = Wdiag_z * WN_factor;
            acados_in.WN[3 + 3*(NX)] = Wdiag_u * WN_factor;
            acados_in.WN[4 + 4*(NX)] = Wdiag_v * WN_factor;
            acados_in.WN[5 + 5*(NX)] = Wdiag_w * WN_factor;
            acados_in.WN[6 + 6*(NX)] = Wdiag_p * WN_factor;
            acados_in.WN[7 + 7*(NX)] = Wdiag_q * WN_factor;
            acados_in.WN[8 + 8*(NX)] = Wdiag_r * WN_factor;
            acados_in.WN[9 + 9*(NX)] = Wdiag_phi * WN_factor;
            acados_in.WN[10 + 10*(NX)] = Wdiag_th * WN_factor;
            acados_in.WN[11 + 11*(NX)] = Wdiag_psi * WN_factor;
            acados_in.WN[12 + 12*(NX)] = Wdiag_gamma * WN_factor;

            // --- Read Estimate from incoming message (adapt fields to your estimator)
            // position
            acados_in.x0[x] = msg->pos.x;
            acados_in.x0[y] = msg->pos.y;
            acados_in.x0[z] = msg->pos.z;

            // For attitude: prefer quaternion if provided, else Euler
            Quaterniond q_est;
            q_est.w() = msg->quat.w;
            q_est.x() = msg->quat.x;
            q_est.y() = msg->quat.y;
            q_est.z() = msg->quat.z;
            q_est.normalize();
            euler eu = quatern2euler(&q_est);

            acados_in.x0[phi] = eu.phi;
            acados_in.x0[th]  = eu.theta;
            acados_in.x0[psi] = eu.psi;

            // body velocities (make sure your estimator populates msg->vel as body-frame velocities u,v,w)
            acados_in.x0[u] = msg->vel.x;
            acados_in.x0[v] = msg->vel.y;
            acados_in.x0[w] = msg->vel.z;

            // rates p,q,r (body angular rates)
            acados_in.x0[p] = msg->rates.x;
            acados_in.x0[q] = msg->rates.y;
            acados_in.x0[r] = msg->rates.z;

            // gamma (flight-path angle) computed from (u,w)
            acados_in.x0[gamma] = compute_gamma(acados_in.x0[u], acados_in.x0[w]);

            // Set bounds: set initial state equality constraints (lbx=ubx=x0)
            ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, 0, "lbx", acados_in.x0);
            ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, 0, "ubx", acados_in.x0);

            // fill yref from acados_in.yref (already filled earlier based on policy)
            for (i = 0; i < N; i++) {
                for (j = 0; j < NY; ++j) acados_in.yref[i*NY + j] = acados_in.yref[i*NY + j]; // already set
                ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "yref", acados_in.yref + i*NY);
            }
            ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, N, "yref", acados_in.yref_e);

            #if SET_WEIGHTS
            for (ii = 0; ii < N; ii++)
            {
                ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, ii, "W", acados_in.W);
            }
            ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, N, "W", acados_in.WN);
            #endif

            // solve
            acados_status = acados_solve();

            // gather results
            acados_out.status = acados_status;
            acados_out.KKT_res = (double)nlp_out->inf_norm_res;
            acados_out.cpu_time = (double)nlp_out->total_time;

            // get first control
            ocp_nlp_out_get(nlp_config, nlp_dims, nlp_out, 0, "u", (void *)acados_out.u0);

            // publish control surfaces as Float32MultiArray: [delta_e, delta_a, delta_r, delta_t]
            std_msgs::Float32MultiArray ctrl;
            ctrl.data.resize(4);
            ctrl.data[0] = acados_out.u0[delta_e];
            ctrl.data[1] = acados_out.u0[delta_a];
            ctrl.data[2] = acados_out.u0[delta_r];
            ctrl.data[3] = acados_out.u0[delta_t];
            p_ctrl_surf.publish(ctrl);

            // also publish simplified "body_cmd" for a lower-level autopilot (angles in degrees, throttle 0..1)
            geometry_msgs::Twist bodycmd;
            bodycmd.linear.x  = rad2Deg(acados_out.u0[delta_e]); // elevator in degrees
            bodycmd.linear.y  = rad2Deg(acados_out.u0[delta_a]); // aileron in degrees
            bodycmd.linear.z  = acados_out.u0[delta_t];          // throttle 0..1
            bodycmd.angular.z = rad2Deg(acados_out.u0[delta_r]); // rudder in degrees
            p_bodycmd.publish(bodycmd);

        } catch (int acados_status) {
            ROS_INFO_STREAM("An exception occurred. Exception Nr. " << acados_status << endl);
        }
    }
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "fixedwing_nmpc");
    ros::NodeHandle n("~");

    std::string ref_traj;
    n.getParam("ref_traj", ref_traj);

    NMPC nmpc(n, ref_traj);
    nmpc.run();

    return 0;
}
