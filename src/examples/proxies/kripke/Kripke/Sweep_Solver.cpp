/*--------------------------------------------------------------------------
 * Sweep-based solver routine.
 *--------------------------------------------------------------------------*/

#include <Kripke.h>
#include <Kripke/Subdomain.h>
#include <Kripke/SubTVec.h>
#include <Kripke/ParallelComm.h>
#include <Kripke/Grid.h>
#include <vector>
#include <stdio.h>
#include <ams/Client.hpp>

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------
 * Begin Ascent Integration
 *--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

#include <conduit_blueprint.hpp>
#include <ascent.hpp>

/* SERVIZ HELPER CODE*/
namespace tl = thallium;

/* Globals -- Yikes, I know. */
static std::string g_address_file;
static std::string g_address;
static std::string g_protocol = "na+sm";
static std::string g_node;
static unsigned    g_provider_id;
static std::string g_log_level = "info";
int use_local = 1;
int i_should_participate_in_server_calls = 0;
int num_server = 1;
tl::engine *engine;
ams::Client *client;
ams::NodeHandle ams_client;
std::vector<tl::async_response> areq_array;
int current_buffer_index = 0;
MPI_Comm new_comm;
int key = 0;
int color = 0;
int new_rank = 0;
int use_partitioning = 0;
int max_step = 0;
int server_instance_id = 0;

/* End globals */

static void parse_command_line();

/* Helper function to read and parse input args */
static std::string read_nth_line(const std::string& filename, int n)
{
   std::ifstream in(filename.c_str());

   std::string s;
   //for performance
   s.reserve(200);

   //skip N lines
   for(int i = 0; i < n; ++i)
       std::getline(in, s);

   std::getline(in,s);
   return s;
}

void parse_command_line() {
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    char *addr_file_name = getenv("AMS_SERVER_ADDR_FILE");
    char *node_file_name = getenv("AMS_NODE_ADDR_FILE");
    char *use_local_opt = getenv("AMS_USE_LOCAL_ASCENT");
    char *num_servers = getenv("AMS_NUM_SERVERS_PER_INSTANCE");
    server_instance_id = std::stoi(std::string(getenv("AMS_SERVER_INSTANCE_ID")));
    max_step = std::stoi(std::string(getenv("AMS_MAX_STEP")));

    /* The logic below grabs the server address corresponding the client's MPI rank (MXM case) */
    std::string num_servers_str = num_servers;
    std::stringstream n_(num_servers_str);
    n_ >> num_server;
    std::string use_local_str = use_local_opt;
    std::stringstream s__(use_local_str);
    s__ >> use_local;

    if(use_local)
        return;

    if(size > num_server) {
        use_partitioning = 1;
        key = rank;
        color = (int)(rank/(size/num_server));
        MPI_Comm_split(MPI_COMM_WORLD, color, key, &new_comm);
        MPI_Comm_rank(new_comm, &new_rank);
        if(new_rank == 0) {
            size_t pos = 0;
            g_address_file = std::string(addr_file_name);
            std::string delimiter = " ";
            std::string l = read_nth_line(g_address_file, server_instance_id*num_server + color + 1);
            pos = l.find(delimiter);
            std::string server_rank_str = l.substr(0, pos);
            std::stringstream s_(server_rank_str);
            int server_rank;
            s_ >> server_rank;
            assert(server_rank == color);
            l.erase(0, pos + delimiter.length());
            g_address = l;
            g_provider_id = 0;
            g_node = read_nth_line(std::string(node_file_name), server_instance_id*num_server + color);
            g_protocol = g_address.substr(0, g_address.find(":"));
            i_should_participate_in_server_calls = 1;
        }
    } else {
        size_t pos = 0;
        g_address_file = std::string(addr_file_name);
        std::string delimiter = " ";
        std::string l = read_nth_line(g_address_file, server_instance_id*num_server + rank + 1);
        pos = l.find(delimiter);
        std::string server_rank_str = l.substr(0, pos);
        std::stringstream s_(server_rank_str);
        int server_rank;
        s_ >> server_rank;
        assert(server_rank == rank);
        l.erase(0, pos + delimiter.length());
        g_address = l;
        g_provider_id = 0;
        g_node = read_nth_line(std::string(node_file_name), server_instance_id*num_server + rank);
        g_protocol = g_address.substr(0, g_address.find(":"));
        i_should_participate_in_server_calls = 1;
    }
}

static double wait_for_pending_requests()
{

    double start = MPI_Wtime();
    for(auto i = areq_array.begin(); i != areq_array.end(); i++) {
        bool ret;
        i->wait();
    }
    double end = MPI_Wtime() - start;
    return end;
}

using namespace conduit;
using ascent::Ascent;
static int count = 0;
static int max_backlog = 0;

void writeAscentData(Ascent &ascent, Grid_Data *grid_data, int timeStep)
{
   static int ams_initialized = 0;
   static double total_time = 0;
   static double total_rpc_time = 0;
   static double total_part_time = 0;
   static double total_barrier_time = 0;
   static double total_allreduce_time = 0;
   conduit::Node ascent_opts;
   ascent_opts["mpi_comm"] = MPI_Comm_c2f(MPI_COMM_WORLD);
   ascent_opts["runtime/type"] = "ascent";

   /* SERVIZ Initialization */
   if(!ams_initialized) {
        /*Connect to server */
        parse_command_line();
	ams_initialized = 1;
	engine = new tl::engine(g_protocol, THALLIUM_CLIENT_MODE);
	client = new ams::Client(*engine);
    	if(!use_local and i_should_participate_in_server_calls) {
    	    // Initialize a Client
            ams_client = (*client).makeNodeHandle(g_address, g_provider_id,
            	ams::UUID::from_string(g_node.c_str()));
    	}
   }

  grid_data->kernel->LTimes(grid_data);
  conduit::Node data;

  int mpi_size;
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  int myid;
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  int num_zone_sets = grid_data->zs_to_sdomid.size();

  // TODO: we don't support domain overloading ...
  for(int sdom_idx = 0; sdom_idx < grid_data->num_zone_sets; ++sdom_idx)
  {
    ASCENT_BLOCK_TIMER(COPY_DATA);

    int sdom_id =  grid_data->zs_to_sdomid[sdom_idx];
    Subdomain &sdom = grid_data->subdomains[sdom_id];
    //create coords array
    conduit::float64 *coords[3];

    data["state/time"]   = (conduit::float64)3.1415;
    data["state/domain_id"] = (conduit::uint64) myid;
    data["state/cycle"]  = (conduit::uint64) timeStep;

    data["state/performance/incomingRequests"] = ParallelComm::getIncomingRequests();
    data["state/performance/outgointRequests"] = ParallelComm::getOutgoingRequests();
    data["state/performance/loops"] = count;
    data["state/performance/max_backlog"] = max_backlog;
    ParallelComm::resetRequests();

    data["coordsets/coords/type"]  = "rectilinear";
    data["coordsets/coords/values/x"].set(conduit::DataType::float64(sdom.nzones[0]+1));
    coords[0] = data["coordsets/coords/values/x"].value();
    data["coordsets/coords/values/y"].set(conduit::DataType::float64(sdom.nzones[1]+1));
    coords[1] = data["coordsets/coords/values/y"].value();
    data["coordsets/coords/values/z"].set(conduit::DataType::float64(sdom.nzones[2]+1));
    coords[2] = data["coordsets/coords/values/z"].value();

    data["topologies/mesh/type"]      = "rectilinear";
    data["topologies/mesh/coordset"]  = "coords";

    for(int dim = 0; dim < 3;++ dim)
    {
      coords[dim][0] = sdom.zeros[dim];
      for(int z = 0;z < sdom.nzones[dim]; ++z)
      {
        coords[dim][1+z] = coords[dim][z] + sdom.deltas[dim][z];
      }
    }
    data["fields/phi/association"] = "element";
    data["fields/phi/topology"] = "mesh";
    data["fields/phi/type"] = "scalar";

    data["fields/phi/values"].set(conduit::DataType::float64(sdom.num_zones));
    conduit::float64 * phi_scalars = data["fields/phi/values"].value();

    // TODO can we do this with strides and not copy?
    for(int i = 0; i < sdom.num_zones; i++)
    {
      phi_scalars[i] = (*sdom.phi)(0,0,i);
    }

  }//each sdom

   //------- end wrapping with Conduit here -------//
  conduit::Node verify_info;
  if(!conduit::blueprint::mesh::verify(data,verify_info))
  {
      CONDUIT_INFO("blueprint verify failed!" + verify_info.to_json());
  }
  else
  {
      CONDUIT_INFO("blueprint verify succeeded");
  }

  conduit::Node actions;
  conduit::Node scenes;
  scenes["s1/plots/p1/type"]         = "volume";
  scenes["s1/plots/p1/field"] = "phi";


  conduit::Node &add_plots = actions.append();
  add_plots["action"] = "add_scenes";
  add_plots["scenes"] = scenes;

  actions.append()["action"] = "execute";
  actions.append()["action"] = "reset";

  /* SERVIZ PARTITIONING */
  double start = MPI_Wtime();

  /* Mesh partitioning using Conduit */
  conduit::Node partitioned_mesh;
  conduit::Node partitioning_options;
  int new_size;

  /* Get timestamp */
  unsigned int ts, min_ts = 0;
  struct timeval tv;
  gettimeofday(&tv, NULL);
  ts = (unsigned int)(tv.tv_sec * 1000 + tv.tv_usec / 1000);

  double start_ts = MPI_Wtime();
  if(!use_local)
    MPI_Allreduce(&ts, &min_ts, 1, MPI_UNSIGNED, MPI_MIN, MPI_COMM_WORLD);
  double end_ts = MPI_Wtime() - start_ts;

  double start_part = MPI_Wtime();

  if(!use_local and use_partitioning and (current_buffer_index % (std::stoi(std::string(getenv("AMS_VIZ_FREQUENCY")))) == 0)) {
      MPI_Comm_size(new_comm, &new_size);
      int new_rank;
      MPI_Comm_rank(new_comm, &new_rank);

      std::string bp_mesh_str = data.to_string("conduit_base64_json");
      int bp_mesh_size = bp_mesh_str.size();

      int * bp_mesh_sizes = (int*)malloc(sizeof(int)*new_size);
      int * displacements = (int*)malloc(sizeof(int)*new_size);
      int total_bytes = 0;

      MPI_Gather(&bp_mesh_size, 1, MPI_INT, bp_mesh_sizes, 1, MPI_INT, 0, new_comm);

      displacements[0] = 0;
      if(new_rank == 0) {
         for (int i = 0; i < new_size; i++) {
             total_bytes += bp_mesh_sizes[i];
             if(i > 0)
                 displacements[i] = displacements[i-1] + bp_mesh_sizes[i-1];
         }
      }
      char * recv_string = (char*)malloc(sizeof(char)*total_bytes);

      MPI_Gatherv(bp_mesh_str.data(), bp_mesh_size, MPI_BYTE, recv_string, bp_mesh_sizes, displacements, MPI_BYTE, 0, new_comm);
      std::vector<std::string> partition_strings;

      if(new_rank == 0) {
         std::vector<const conduit::Node *> doms;
         std::vector<conduit::Node> meshes;
         std::vector<conduit::index_t> chunk_ids;
         conduit::Node test_mesh;
         for(int i = 0; i < new_size; i++) {
           conduit::Node mesh_to_be_partitioned;
           partition_strings.push_back(std::string(recv_string+displacements[i], bp_mesh_sizes[i]));
           mesh_to_be_partitioned.parse(partition_strings[partition_strings.size()-1], "conduit_base64_json");
           meshes.push_back(std::move(mesh_to_be_partitioned));
         }
         for(int i = 0; i < meshes.size(); i++) {
           partitioned_mesh.update(meshes[i]);
         }
      }
  }
      
  double end_part = MPI_Wtime() - start_part;
  /* RPC or local in-situ */
  double start_rpc = MPI_Wtime();
  if(!use_local and i_should_participate_in_server_calls and (current_buffer_index % (std::stoi(std::string(getenv("AMS_VIZ_FREQUENCY")))) == 0)) {
     if(use_partitioning) {
        auto response = ams_client.ams_open_publish_execute(ascent_opts, partitioned_mesh, 0, actions, min_ts);
        areq_array.push_back(std::move(response));
     } else {
        auto response = ams_client.ams_open_publish_execute(ascent_opts, data, 0, actions, min_ts);
        areq_array.push_back(std::move(response));
     }
  } else if(use_local and (current_buffer_index % (std::stoi(std::string(getenv("AMS_VIZ_FREQUENCY")))) == 0)) {
       	ascent.publish(data);
        ascent.execute(actions);
  }

  double end_rpc = MPI_Wtime() - start_rpc;
  double end = MPI_Wtime();
  total_time += end-start;
  total_rpc_time += end_rpc;
  total_part_time += end_part;
  total_allreduce_time += end_ts;
  if(myid == 0) {
	std::cerr << "DATA SIZE: " << (data.to_string()).size() << std::endl;
       	std::cout << "======================================================" << std::endl;
       	std::cout << "Current Total time: " << total_time  << std::endl;
       	std::cout << "Current Total partitioning cost: " << total_part_time << std::endl; 
       	std::cout << "Current Total RPC time: " << total_rpc_time << std::endl;
       	std::cout << "======================================================" << std::endl;
  }

  current_buffer_index += 1;
  /* Before I exit, checking for pending requests sitting around */
  if(!use_local and current_buffer_index == max_step + 1) {
      double wait_time = wait_for_pending_requests();
      double max_total_time, max_total_rpc_time, max_total_part_time, max_total_mesh_time, max_total_allreduce_time, max_total_wait_time;
      MPI_Barrier(MPI_COMM_WORLD);

      MPI_Reduce(&total_time, &max_total_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      MPI_Reduce(&total_rpc_time, &max_total_rpc_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      MPI_Reduce(&total_part_time, &max_total_part_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      MPI_Reduce(&total_allreduce_time, &max_total_allreduce_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      MPI_Reduce(&wait_time, &max_total_wait_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

      if(myid == 0) {
          std::cerr << "Task ID: " << std::stoi(std::string(getenv("AMS_TASK_ID"))) << " is done." << std::endl;
          std::cout << "======================================================" << std::endl;
          std::cout << "Total time: " << max_total_time  << std::endl;
          std::cout << "Total partitioning cost: " << max_total_part_time << std::endl; 
          std::cout << "Total RPC time: " << max_total_rpc_time << std::endl;
          std::cout << "Total timestamp collection time: " << max_total_allreduce_time << std::endl;
          std::cout << "Total wait time: " << max_total_wait_time << std::endl;
          std::cout << "======================================================" << std::endl;
      }
      if(i_should_participate_in_server_calls and (std::stoi(std::string(getenv("AMS_TASK_ID"))) == (std::stoi(std::string(getenv("AMS_MAX_TASK_ID")))) and (std::stoi(std::string(getenv("AMS_SERVER_MODE"))) == 1)))  {
           ams_client.ams_execute_pending_requests();
      }
      MPI_Barrier(MPI_COMM_WORLD);
      margo_instance_id mid = engine->get_margo_instance();
      margo_finalize(mid);
  }

}

/**
  Run solver iterations.
*/
int SweepSolver (Grid_Data *grid_data, bool block_jacobi)
{
  conduit::Node ascent_opts;
  ascent_opts["mpi_comm"] = MPI_Comm_c2f(MPI_COMM_WORLD);
  ascent_opts["runtime/type"] = "ascent";

  Ascent ascent;
  ascent.open(ascent_opts);

  conduit::Node testNode;
  Kernel *kernel = grid_data->kernel;

  int mpi_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

  double TOTAL_START = MPI_Wtime();

  BLOCK_TIMER(grid_data->timing, Solve);
  {

  // Loop over iterations
  double part_last = 0.0;
 for(int iter = 0;iter < grid_data->niter;++ iter){

   {//ascent block timer
     ASCENT_BLOCK_TIMER(KRIPKE_MAIN_LOOP);
    /*
     * Compute the RHS:  rhs = LPlus*S*L*psi + Q
     */

    // Discrete to Moments transformation (phi = L*psi)
    {
      BLOCK_TIMER(grid_data->timing, LTimes);
      kernel->LTimes(grid_data);
    }

    // Compute Scattering Source Term (psi_out = S*phi)
    {
      BLOCK_TIMER(grid_data->timing, Scattering);
      kernel->scattering(grid_data);
    }

    // Compute External Source Term (psi_out = psi_out + Q)
    {
      BLOCK_TIMER(grid_data->timing, Source);
      kernel->source(grid_data);
    }

    // Moments to Discrete transformation (rhs = LPlus*psi_out)
    {
      BLOCK_TIMER(grid_data->timing, LPlusTimes);
      kernel->LPlusTimes(grid_data);
    }

    /*
     * Sweep (psi = Hinv*rhs)
     */
    {
      BLOCK_TIMER(grid_data->timing, Sweep);

      if(true){
        // Create a list of all groups
        std::vector<int> sdom_list(grid_data->subdomains.size());
        for(int i = 0;i < grid_data->subdomains.size();++ i){
          sdom_list[i] = i;
        }

        // Sweep everything
        SweepSubdomains(sdom_list, grid_data, block_jacobi);
      }
      // This is the ARDRA version, doing each groupset sweep independently
      else{
        for(int group_set = 0;group_set < grid_data->num_group_sets;++ group_set){
          std::vector<int> sdom_list;
          // Add all subdomains for this groupset
          for(int s = 0;s < grid_data->subdomains.size();++ s){
            if(grid_data->subdomains[s].idx_group_set == group_set){
              sdom_list.push_back(s);
            }
          }

          // Sweep the groupset
          SweepSubdomains(sdom_list, grid_data, block_jacobi);
        }
      }
    }
   }//end main loop timing
    double part = grid_data->particleEdit();
    writeAscentData(ascent, grid_data, iter);
    if(mpi_rank==0){
      printf("iter %d: particle count=%e, change=%e\n", iter, part, (part-part_last)/part);
    }
    part_last = part;
  }

  ascent.close();
  }//Solve block

  double TOTAL_END = MPI_Wtime() - TOTAL_START;
  if(mpi_rank == 0) {
      fprintf(stdout, "TOTAL EXECUTION TIME: %lf\n", TOTAL_END);
  }

  //Ascent: we don't want to execute all loop orderings, so we will just exit;
  MPI_Finalize();
  exit(0);
  return(0);
}

/*  --------------------------------------------------------------------------
 *  --------------------------------------------------------------------------
 *   End Ascent Integration
 *  --------------------------------------------------------------------------
 *  --------------------------------------------------------------------------*/


/**
  Perform full parallel sweep algorithm on subset of subdomains.
*/
void SweepSubdomains (std::vector<int> subdomain_list, Grid_Data *grid_data, bool block_jacobi)
{
  // Create a new sweep communicator object
  ParallelComm *comm = NULL;
  if(block_jacobi){
    comm = new BlockJacobiComm(grid_data);
  }
  else {
    comm = new SweepComm(grid_data);
  }

  // Add all subdomains in our list
  for(int i = 0;i < subdomain_list.size();++ i){
    int sdom_id = subdomain_list[i];
    comm->addSubdomain(sdom_id, grid_data->subdomains[sdom_id]);
  }
  count = 0;
  max_backlog = 0;
  /* Loop until we have finished all of our work */
  while(comm->workRemaining()){
    count++;
    // Get a list of subdomains that have met dependencies
    std::vector<int> sdom_ready = comm->readySubdomains();
    int backlog = sdom_ready.size();
    max_backlog = max_backlog < backlog ? backlog : max_backlog;
    // Run top of list
    if(backlog > 0){
      int sdom_id = sdom_ready[0];
      Subdomain &sdom = grid_data->subdomains[sdom_id];
      // Clear boundary conditions
      for(int dim = 0;dim < 3;++ dim){
        if(sdom.upwind[dim].subdomain_id == -1){
          sdom.plane_data[dim]->clear(0.0);
        }
      }
      {
        BLOCK_TIMER(grid_data->timing, Sweep_Kernel);
        // Perform subdomain sweep
        grid_data->kernel->sweep(&sdom);
      }

      // Mark as complete (and do any communication)
      comm->markComplete(sdom_id);
    }
  }
  delete comm;
}


