/*
 * particle_filter.cpp
 *
 *  Created on: Sept 07, 2018
 *      Author: Allay Desai
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) 
{
	
	//Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) 
	if(!is_initialized)
	{
		// Create gaussian noise
		default_random_engine gen;
		normal_distribution<double> dist_x(x, std[0]);
		normal_distribution<double> dist_y(y, std[1]);
		normal_distribution<double> dist_theta(theta, std[2]);

		// Initialize all particles
		num_particles = 100;

		// Set the number of particles. 
		particles.resize(num_particles);

		for (int i=0; i<num_particles; i++)
		{
			particles[i].id = i;
			// Add random Gaussian noise to each particle.
			particles[i].x = dist_x(gen);
			particles[i].y = dist_y(gen);
			particles[i].theta = dist_theta(gen);
			particles[i].weight = 1.0;
		}

		// Initialize all weights to 1. 
		weights.resize(num_particles);


		for (int i=0; i<num_particles; i++)
		{
			weights[i] = 1.0;
		}
		
		// Set initialized Bool
		is_initialized = true;
	}
	else
	{
		return;
	}
	
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) 
{
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	// Create gaussian noise
	default_random_engine gen;
	normal_distribution<double> dist_x(0, std_pos[0]);
	normal_distribution<double> dist_y(0, std_pos[1]);
	normal_distribution<double> dist_theta(0, std_pos[2]);

	// Add measurements to each particle
	for (int i=0; i<num_particles; i++)
	{
		// Check if yaw rate close to zero
		if(abs(yaw_rate) < 0.0001)
		{
			particles[i].x += velocity * cos(particles[i].theta) * delta_t;
			particles[i].y += velocity * sin(particles[i].theta)* delta_t;
		}
		else
		{
			particles[i].x += (velocity / yaw_rate) * (sin(particles[i].theta + (yaw_rate * delta_t)) - sin(particles[i].theta));
			particles[i].y += (velocity / yaw_rate) * (cos(particles[i].theta) - cos(particles[i].theta + (yaw_rate * delta_t)));
			particles[i].theta += yaw_rate * delta_t;
		}

		// Add random Gaussian noise
		particles[i].x += dist_x(gen);
		particles[i].y += dist_y(gen);
		particles[i].theta += dist_theta(gen);
	}

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// Update the weights of each particle using a mult-variate Gaussian distribution. 
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	// according to the MAP'S coordinate system. You will need to transform between the two systems.
	// Keep in mind that this transformation requires both rotation AND translation (but no scaling).

	for (int i=0; i<num_particles; i++)
	{
		vector<LandmarkObs> conv_observations;
		
		vector<int> associations;
		vector<double> sense_x;
		vector<double> sense_y;
		
		particles[i].weight = 1.0;

		// Transform Car observations to map coordinates
		for (int j=0; j<observations.size(); j++)
		{
			LandmarkObs conv_obs;
			LandmarkObs raw_obs;
			
			raw_obs = observations[j];

			// Space transformation from vehicle to map coordinates
			conv_obs.x = particles[i].x + (raw_obs.x * cos(particles[i].theta) - raw_obs.y * sin(particles[i].theta));
			conv_obs.y = particles[i].y + (raw_obs.x * sin(particles[i].theta) + raw_obs.y * cos(particles[i].theta));
			
			conv_observations.push_back(conv_obs);
		}
		
		// Check Association 	
		for (int j=0; j<conv_observations.size(); j++)
		{
			// Default max sensor range
			double min_dist = sensor_range;
			int assoc_lmark = -1;
			// Look through list of landmarks
			for (int k = 0; k < map_landmarks.landmark_list.size(); k++)
			{
				double landmark_x = map_landmarks.landmark_list[k].x_f;
				double landmark_y = map_landmarks.landmark_list[k].y_f;
				double x_diff = conv_observations[j].x - landmark_x;
				double y_diff = conv_observations[j].y - landmark_y;
				// Calc distance from landmark
				double calc_dist = sqrt(pow(x_diff, 2.0) + pow(y_diff, 2.0));
				// Compare with current minimum distance
				if (calc_dist < min_dist)
				{
					// New min distance landmark found
					min_dist = calc_dist;
					assoc_lmark = k;
				}
			}
			if (assoc_lmark!=0)
			{
				// update weights
				double o_x = conv_observations[j].x;
				double o_y = conv_observations[j].y;
				double l_x = map_landmarks.landmark_list[assoc_lmark].x_f;
				double l_y = map_landmarks.landmark_list[assoc_lmark].y_f;
				double ol_x_diff = o_x-l_x;
				double ol_y_diff = o_y-l_y;
				// Calc product term
				long double product = 1 / (2 * M_PI * std_landmark[0] * std_landmark[1]) 
									* exp(-(pow(ol_x_diff, 2.0) * 0.5 / pow(std_landmark[0], 2.0) + pow(ol_y_diff, 2.0) * 0.5 / pow(std_landmark[1], 2.0)));
				// Check to see if product term very small
				if (product > 0.0001)
				{
					particles[i].weight *= product;
				}
				else
				{
					particles[i].weight *= 0.00001;
				}
			}
			// Append to list of associations and sensor values
			associations.push_back(assoc_lmark + 1);
			sense_x.push_back(conv_observations[j].x);
			sense_y.push_back(conv_observations[j].y);
		}
		// Update each particle with respective associations, weight and sensor value
		particles[i] = SetAssociations(particles[i], associations, sense_x, sense_y);
		weights[i] = particles[i].weight;
	}
}

void ParticleFilter::resample() 
{
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	// http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	default_random_engine gen;
    discrete_distribution<int> dist_w(weights.begin(), weights.end());

    vector<Particle> resample_particles(num_particles);

  	for (int i = 0; i < num_particles; i++)
    {
        resample_particles[i] = particles[dist_w(gen)];
    }
	
	particles = move(resample_particles);

}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations = associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;

    return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
