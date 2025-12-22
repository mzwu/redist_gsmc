#include <algorithm>
#include <set>
#include <RcppThread.h>
#include "smc_base.h"
#include "tree_op.h"


#ifndef MAP_CALC_H
#define MAP_CALC_H



/*
 * Compute the Fryer-Holden penalty for district `distr`
 */
double eval_fry_hold(const subview_col<uword> &districts, int distr,
                     const uvec &total_pop, mat ssdmat, double denominator);





/*
 * Compute the qps penalty for district `distr`
 */
double eval_qps(const subview_col<uword> &districts, int distr,
                const uvec &total_pop, const uvec &cities, int n_city,
                int nd);

/*
 * Compute the log spanning tree penalty for district `distr`
 */
double eval_log_st(const subview_col<uword> &districts, const Graph g,
                   arma::uvec counties, int ndists);

/*
 * Compute the log spanning tree penalty for district `distr`
 */
double eval_er(const subview_col<uword> &districts, const Graph g, int ndists);



/*
 * Compute the cooccurence matrix for a set of precincts indexed by `idxs`,
 * given a collection of plans
 */
// [[Rcpp::export]]
arma::mat prec_cooccur(arma::umat m, arma::uvec idxs, int ncores=0);

/*
 * Compute the percentage of `group` in each district. Asummes `m` is 1-indexed.
 */
// [[Rcpp::export]]
NumericMatrix group_pct(IntegerMatrix const &plans_mat, 
    arma::vec const &group_pop, arma::vec const &total_pop, 
    int const n_distr, int const ncores = 0);

/*
 * Tally a variable by district.
 */
// [[Rcpp::export]]
NumericMatrix pop_tally(IntegerMatrix const &districts, arma::vec const &pop, int const n_distr,
                        int const ncores = 0);


/*
 * Infer the number of seats of the regions 
 */
// [[Rcpp::export]]
Rcpp::IntegerMatrix infer_region_seats(
    Rcpp::IntegerMatrix const &region_pops,
    double const lower, double const upper,
    int const total_seats,
    int const num_threads = 0
);

/*
 * Compute the maximum deviation from the equal population constraint.
 */
// [[Rcpp::export]]
Rcpp::NumericVector max_dev(
    const Rcpp::IntegerMatrix &districts, const arma::vec &pop, int const n_distr,
    bool const multimember_districts = false, int const nseats = -1, Rcpp::IntegerMatrix const &seats_matrix = Rcpp::IntegerMatrix(1,1),
    int const num_threads = 1
);



// computes log number of spanning trees on region intersect county
// In either a region or a merged region 
double compute_log_region_and_county_spanning_tree(
    Graph const &g, const uvec &counties, int const county,
    PlanVector const &region_ids,
    int const region1_id, int const region2_id
);


/*
 * Compute the log number of spanning trees for the contracted (ie county level) graph
 */
double compute_log_county_level_spanning_tree(
    Graph const &g, const uvec &counties, int const n_cty,
    PlanVector const &region_ids,
    int const region1_id, int const region2_id
);




// [[Rcpp::export]]
Rcpp::NumericVector order_district_stats(
    Rcpp::NumericVector const &district_stats, 
    int const ndists,
    int const num_threads
);



// [[Rcpp::export]]
Rcpp::DataFrame order_columns_by_district(
    Rcpp::DataFrame const &df,
    Rcpp::CharacterVector const &columns,
    int const ndists,
    int const num_threads = 0);



/*
 * Non-templated constraint functions
 */

double compute_log_pop_temper(
    double const target, double const pop_temper, int const ndists,
    int const region_pop, int const region_size
);


/************************ 
 * Templated Constraint Functions
 *************************/

/*
 * Compute the population penalty for district `distr`
 */
template <typename PlanID>
double eval_pop_dev(const PlanID &region_ids, 
    int const region1, int const region2,
    arma::uvec const &total_pop, double const parity
) {
    double pop = 0.0;

    for (size_t i = 0; i < total_pop.size(); ++i) {
        if (region_ids[i] == region1 || region_ids[i] == region2) {
            pop += total_pop[i];
        }
    }

    double frac = pop / parity;
    return std::pow(frac - 1.0, 2.0);
}


/*
 * Compute the power-based group penalty for district `distr`
 */
template <typename PlanID>
double eval_grp_pow(
    const PlanID &region_ids, int const V,
    int const region1_id, int const region2_id,
    arma::uvec const &grp_pop, arma::uvec const &total_pop,
    double const tgt_grp, double const tgt_other, double const pow) {
    double sum_grp = 0.0;
    double sum_total = 0.0;

    for (size_t i = 0; i < V; ++i) {
        if (region_ids[i] == region1_id || region_ids[i] == region2_id) {
            sum_grp += grp_pop[i];
            sum_total += total_pop[i];
        }
    }

    if (sum_total == 0.0) return 0.0;  // avoid div-by-zero

    double frac = sum_grp / sum_total;
    return std::pow(std::fabs(frac - tgt_grp) * std::fabs(frac - tgt_other), pow);
}

/*
 * Compute the new, hinge group penalty for district `distr`
 *
 */
template <typename PlanID>
double eval_grp_hinge(
    PlanID const &region_ids, int const V, int const region1_id, int const region2_id,
    arma::vec const &tgts_grp, const arma::uvec &grp_pop, const arma::uvec &total_pop) {
    double subsetted_grp_pop_sum = 0.0;
    double subsetted_total_pop_sum = 0.0;
    // get the sum of the two columns in region 1 or 2
    for (size_t i = 0; i < V; i++)
    {
        auto const region_id = region_ids[i];
        if(region_id == region1_id || region_id == region2_id){
            subsetted_grp_pop_sum += grp_pop[i];
            subsetted_total_pop_sum += total_pop[i];
        }
    }
    // do subsetted_grp_pop_sum/subsetted_total_pop_sum
    double frac = std::exp(
        std::log(subsetted_grp_pop_sum) - std::log(subsetted_total_pop_sum)
    );
    // REprintf("Frac %f\n", frac);
    // figure out which to compare it to
    double target;
    double diff = 1;
    int n_tgt = tgts_grp.size();
    for (int i = 0; i < n_tgt; i++) {
        double new_diff = std::fabs(tgts_grp[i] - frac);
        // REprintf("Target %d - Diff %f, Target %f \n", i, new_diff, tgts_grp[i]);
        if (new_diff <= diff) {
            diff = new_diff;
            target = tgts_grp[i];
        }
    }
    // REprintf("%f\n", target - frac);

    return std::sqrt(std::max(0.0, target - frac));
}


/*
 * Compute the incumbent-preserving penality  
 */
template <typename PlanID>
double eval_inc(
    PlanID const &region_ids, 
    int const region1_id, int const region2_id,
    const arma::uvec &incumbents
){
    int n_inc = incumbents.size();
    double inc_in_distr = -1.0; // first incumbent doesn't count
    for (int i = 0; i < n_inc; i++) {
        if (region_ids[incumbents[i] - 1] == region1_id || region_ids[incumbents[i] - 1] == region2_id)
            inc_in_distr++;
    }

    if (inc_in_distr < 0.0) {
        inc_in_distr = 0.0;
    }

    return inc_in_distr;
};


/*
 * Compute the status quo penalty for district `distr`, maybe needs to be revised ... 
 */
template <typename PlanID>
double eval_sq_entropy(
    PlanID const &region_ids, 
    arma::uvec const &current,
    int const region1_id, int const region2_id, 
    arma::uvec const &pop, 
    int const ndists, int const n_current, int const V
){
    double accuml = 0;
    for (int j = 0; j < n_current; j++) { // 0-indexed districts, last one is remainder
        double pop_overlap = 0;
        double pop_total = 0;
        for (int k = 0; k < V; k++) {
            if (current[k] != j) continue;
            pop_total += pop[k];

            if (region_ids[k] == region1_id || region_ids[k] == region2_id)
                pop_overlap += pop[k];
        }
        double frac = pop_overlap / pop_total;
        if (frac > 0)
            accuml += frac * std::log(frac);
    }

    return -accuml / ndists / std::log(n_current);
}



/*
 * Compute the phase commute penalty for district `distr`
 *
 */
template <typename PlanID>
double eval_phase_commute_gsmc_version(
    PlanID const &region_ids, 
    const arma::uvec &current, 
    const arma::mat &commute_times,
    arma::uvec const &pop,
    int const V, int const region_id) {
    
    double reassigned_pop = 0.0;

    for (int k = 0; k < V; k++) {
        if (region_ids[k] != region_id) continue; // only evaluate blocks in proposed district

        // get old and new districts of current block
        int school_old_idx = current[k] - 1;
        int school_new_idx = region_ids[k];
        
        // compute and compare commute distances to old and new schools
        double commute_old = commute_times(k, school_old_idx);
        double commute_new = commute_times(k, school_new_idx);
        if (commute_old < commute_new) {
            double to_add = pop[k] * (commute_new - commute_old);
            reassigned_pop += to_add;
        }
    }

    // return log(1 + average extra commute time per person in the district)
    double district_pop = 0.0;
    for (size_t i = 0; i < region_ids.size(); ++i) {
        if (region_ids[i] == static_cast<RegionID>(region_id)) {
            district_pop += pop[i];
        }
    }
    double avg_extra = (district_pop > 0.0) ? (reassigned_pop / district_pop) : 0.0;
    return std::log1p(avg_extra);
}



/*
 * Compute the max commute penalty for district `distr`
 *
 */
template <typename PlanID>
double eval_max_commute_gsmc_version(
    PlanID const &region_ids, 
    const arma::uvec &current, 
    const arma::mat &commute_times,
    arma::uvec const &pop,
    int const V, int const region_id) {

    double max_extra = 0.0;

    for (int k = 0; k < V; k++) {
        if (region_ids[k] != region_id) continue; // only evaluate blocks in proposed district

        // get old and new districts of current block
        int school_old_idx = current[k] - 1;
        int school_new_idx = region_ids[k];

        // compute and compare commute distances to old and new schools
        double commute_old = commute_times(k, school_old_idx);
        double commute_new = commute_times(k, school_new_idx);
        double commute_extra = commute_new - commute_old;
        if (commute_extra > max_extra) {
            max_extra = commute_extra;
        }
    }

    // return log(1 + max extra commute time for a person in the district)
    return std::log1p(max_extra);
}



/*
 * Compute the split feeders penalty for district `distr`
 *
 */
template <typename PlanID>
double eval_split_feeders_gsmc_version(
    PlanID const &region_ids, 
    const arma::uvec &lower, 
    const arma::uvec &schools, 
    arma::uvec const &pop,
    int const V, int const region_id) {
    
    // Which lower level districts are sending students to the current upper level district? How many?
    // Key: lower level district ID, Value: number of students that lower level district sent to the current upper level district
    std::unordered_map<int, int> lower_students;
    for (int v = 0; v < V; ++v) {
        // Skip if not in current upper district
        if (region_ids[v] != region_id) continue;

        // Tally population from each lower district
        if (lower_students.find(lower(v)) == lower_students.end()) {
            lower_students[lower(v)] = 0;
        }
        lower_students[lower(v)] += pop(v);
    }

    // For each lower level school sending students, check if less than 25% 
    // of students are going to the current upper level district
    int split_feeders_count = 0;
    for (auto const &pair : lower_students) {
        int lower_distr = pair.first;
        uvec rows_in_lower = find(lower == lower_distr);
        int lower_total = sum(pop(rows_in_lower));
        double frac = (double) pair.second / lower_total;
        if (frac < 0.25) {
            split_feeders_count += 1;
        }
    }

    return split_feeders_count;
}



/*
 * Compute the capacity penalty for district `distr`
 *
 */
template <typename PlanID>
double eval_capacity_gsmc_version(
    PlanID const &region_ids,  
    const arma::uvec &schools, 
    const arma::uvec &schools_capacity, 
    arma::uvec const &pop,
    int const V, int const region_id) {

    // Count how many people are assigned to current district
    int pop_assigned = 0;
    for (int v = 0; v < V; ++v) {
        if (region_ids[v] == region_id) {
            pop_assigned += pop(v);
        }
    }
    
    // What is the capacity of the current district?
    // Assume schools/schools_capacity are in ascending district ID order
    double pop_capacity = schools_capacity(region_id);

    // Estimate number of school-aged children in the district
    double normalized_pop = pop_assigned / sum(pop) * sum(schools_capacity);
    // Calculate and compare ratio
    double ratio = normalized_pop / pop_capacity;

    if (ratio < 0.85 || ratio > 1.15) {
        return 2;
    }
    else if ((0.85 <= ratio && ratio <= 0.94) || 1.05 <= ratio <= 1.14) {
        return 1;
    }
    else if (0.95 <= ratio && ratio <= 1.04) {
        return 0;
    }
}



// helper function
// calculates districts which appear in each county (but not zeros)
template <typename PlanID>
std::vector<std::set<int>> calc_county_dist(
    PlanID const &region_ids, arma::uvec const &counties, 
    int const n_cty, bool const zero_ok) {
    std::vector<std::set<int>> county_dist(n_cty);
    int V = counties.size();
    for (int i = 0; i < n_cty; i++) {
        county_dist[i] = std::set<int>();
    }
    for (int i = 0; i < V; i++) {
        if (zero_ok || region_ids[i] > 0) {
            county_dist[counties[i]-1].insert(region_ids[i]);
        }
    }
    return county_dist;
}


/*
 * Compute the county split penalty for region region1 
 * No merged version because I'm not sure what exactly its doing so instead
 * the region constraint wrapper just duplicates the plan id and makes one 
 * where the regions are actually merged 
 */
template <typename PlanID>
double eval_splits(PlanID const &region_ids, int const region_id,
                   arma::uvec const &admin_units, int const n_admin_units, bool const smc) {
    std::vector<std::set<int>> county_dist = calc_county_dist(region_ids, admin_units, n_admin_units, region_id == 0);

    int splits = 0;
    for (int i = 0; i < n_admin_units; i++) {
        int cty_n_distr = county_dist[i].size();
        // for SMC, just count the split when it crosses the threshold
        // for MCMC there is no sequential nature
        bool cond = smc ? cty_n_distr == 2 : cty_n_distr >= 2;
        if (cond) {
            if (smc) {
                auto search = county_dist[i].find(region_id);
                if (search != county_dist[i].end()) {
                    splits++;
                }
            } else {
                splits++;
            }
        }
    }

    return splits;
}



/*
 * Compute the county multisplit penalty for region `region_id`
 * No merged version because I'm not sure what exactly its doing so instead
 * the region constraint wrapper just duplicates the plan id and makes one 
 * where the regions are actually merged 
 */
template <typename PlanID>
double eval_multisplits(
    PlanID const &region_ids, int const region_id,
    const arma::uvec &admin_units, int const n_admin_units, bool const smc) {
    std::vector<std::set<int>> county_dist = calc_county_dist(region_ids, admin_units, n_admin_units, region_id == 0);

    double splits = 0;
    for (int i = 0; i < n_admin_units; i++) {
        int cty_n_distr = county_dist[i].size();
        // for SMC, just count the split when it crosses the threshold
        // for MCMC there is no sequential nature
        bool cond = smc ? cty_n_distr == 3 : cty_n_distr >= 3;
        if (cond) {
            if (smc) {
                auto search = county_dist[i].find(region_id);
                if (search != county_dist[i].end()) {
                    splits++;
                }
            } else {
                splits++;
            }
        }
    }

    return splits;
}



/*
 * Compute the total splits penalty for region `region_id`
 * No merged version because I'm not sure what exactly its doing so instead
 * the region constraint wrapper just duplicates the plan id and makes one 
 * where the regions are actually merged 
 */
template <typename PlanID>
double eval_total_splits(
    PlanID const &region_ids, int const region_id,
    arma::uvec const &admin_units, int const n_admin_units, bool const smc
) {
    std::vector<std::set<int>> county_dist = calc_county_dist(region_ids, admin_units, n_admin_units, region_id == 0);

    double splits = 0;
    for (int i = 0; i < n_admin_units; i++) {
        int cty_n_distr = county_dist[i].size();
        // no over-counting since every split counts
        if (cty_n_distr > 1) {
            if (smc) {
                auto search = county_dist[i].find(region_id);
                if (search != county_dist[i].end()) {
                    splits++;
                }
            } else {
                splits++;
            }
        }
    }

    return splits;
}


/*
 * Compute the Polsby Popper penalty for region `region1`
 * or the region formed by merging region1 and region2 if region1 != region2
 */
template <typename PlanID>
double eval_polsby(
    PlanID const &region_ids, 
    int const region1_id, int const region2_id,
    int const V, 
    arma::ivec const &from, arma::ivec const &to,
    arma::vec const &area, arma::vec const &perimeter
) {
    double tot_area = 0.0;
    double tot_perim = 0.0;
    constexpr double pi4 = 4.0 * 3.14159265;

    // Sum area directly for vertices in the district
    for (int v = 0; v < V; ++v) {
        if (region_ids[v] == region1_id || region_ids[v] == region2_id) {
            tot_area += area[v];
        }
    }

    // Sum perimeter contributions from boundary edges
    int E = to.n_elem;
    for (int e = 0; e < E; ++e) {
        auto out_vertex = to[e] - 1;
        if (region_ids[out_vertex] == region1_id || region_ids[out_vertex] == region2_id) {
            int src_vertex = from[e] - 1;
            // only count if its on the permiter
            // need to subtract one because R is 1 indexed
            if (src_vertex <=-1 || (region_ids[src_vertex] != region1_id && region_ids[src_vertex] != region2_id)) {
                tot_perim += perimeter[e];
            }
        }
    }

    double dist_peri2 = std::pow(tot_perim, 2.0);

    return 1.0 - (pi4 * tot_area / dist_peri2);

}




/*
 * Compute the segregation penalty for district `distr`
 */
template <typename PlanID>
double eval_segregation(
    const PlanID &region_ids, 
    int const region1_id, int const region2_id,
    int const V, 
    const arma::uvec &grp_pop, const arma::uvec &total_pop) {
    // Step 1: compute overall group share (pAll) and total population
    double total_grp = arma::sum(grp_pop);
    double total_pop_sum = arma::sum(total_pop);

    if (total_pop_sum == 0.0) return 0.0;

    double pAll = total_grp / total_pop_sum;
    double denom = 2.0 * total_pop_sum * pAll * (1.0 - pAll);
    if (denom == 0.0) return 0.0;

    // Step 2: sum population and group population in the target district
    double local_grp = 0.0;
    double local_pop = 0.0;

    for (int i = 0; i < V; ++i) {
        // count if in either region
        if (region_ids[i] == region1_id || region_ids[i] == region2_id) {
            local_grp += grp_pop[i];
            local_pop += total_pop[i];
        }
    }

    if (local_pop == 0.0) return 0.0;

    // Step 3: compute and return segregation score
    return local_pop * std::abs((local_grp / local_pop) - pAll) / denom;
}



#endif
