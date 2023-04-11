using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using AustinHarris.JsonRpc;


public class MyVector3{
    public float x;
    public float y;
    public float z;

    public MyVector3(Vector3 v){
        x = v.x;
        y = v.y;
        z = v.z;
    }

    public Vector3 AsVector3(){
        return new Vector3(x,y,z);
    }
}

public class ImageObs{
    public int[,,] obs;

    public ImageObs(Color32[] color_image, int width, int height){

        obs = new int[height, width,1];

        for (int i = 0; i < height; i++)
        {
            for (int j = 0; j < width; j++)
            {
                // for (int k = 0; k < 3; k++){
                    obs[height-1-i, j, 0] = (color_image[i * width + j][0]+color_image[i * width + j][1]+color_image[i * width + j][2])/3;
                // }
            }
        }
    }
}

public class RlReset {
    public float total_reward_available;
    public int[,,] obs;

    public RlReset(float total_reward_available,Color32[] color_image, int width, int height){
        int[,,] observation = new int[height, width,1];

        for (int i = 0; i < height; i++)
        {
            for (int j = 0; j < width; j++)
            {
                // for (int k = 0; k < 3; k++){
                    observation[height-1-i, j, 0] = (color_image[i * width + j][0]+color_image[i * width + j][1]+color_image[i * width + j][2])/3;
                // }
            }
        }

        this.total_reward_available = total_reward_available;
        this.obs = observation;
    }
}

public class RlResult {
    public float reward;
    public bool done;
    public int[,,] obs;

    public RlResult(float reward, bool done,ImageObs obs){
        this.reward = reward;
        this.done = done;
        this.obs = obs.obs;
    
    }
}

public class Agent : MonoBehaviour
{
    public float move_Speed = 30f;
    public float rotate_Speed = 1000f;

    private float x;
    private float z;

    public GameObject alga_prefab;
    public int alga_resolution = 2; //2 alga per unit 
    public float boat_size = 10f;
    private float boat_size_x;
    private float boat_size_z;
    public GameObject boat;

    private float res_inverse;
    private Vector3 spawn_position;
    private GameObject spawner;

    Rpc rpc;
    Simulation simulation;

    int alga_remaining;
    int alga_init_count;
    float reward;
    bool done;

    class Rpc : JsonRpcService{

        Agent agent;

        public Rpc(Agent agent){
            this.agent = agent;
        }

        [JsonRpcMethod]
        void Say(string message){
            Debug.Log($"You sent {message}");
        }

        [JsonRpcMethod]
        MyVector3 GetPos(){
            return new MyVector3(agent.transform.position);
        }

        [JsonRpcMethod]
        void Translate(MyVector3 translate){
            agent.transform.position += translate.AsVector3();
        }

        [JsonRpcMethod]
        RlResult Step(int action){  //nothing:0, forward:1, backward:2, left:3, right:4
            return agent.Step(action);
        }

        [JsonRpcMethod]
        RlReset Reset(){  //nothing:0, forward:1, backward:2, left:3, right:4
            return agent.Reset();
        }
    }

    private RenderTexture renderTexture;
    public Camera cam;
    public int resWidth = 128;
    public int resHeight = 128;
    private int step;
    public int max_step = 1000;

    // Start is called before the first frame update
    void Start()
    {
        simulation = GetComponent<Simulation>();

        if(cam.targetTexture == null){
            cam.targetTexture = new RenderTexture(resWidth, resHeight,0);
        }

        rpc = new Rpc(this);
    }

    public RlResult Step(int action){
        reward = 0;

        switch(action){
            case 0:
                break;
            case 1:
                this.transform.position += transform.forward * move_Speed * simulation.SimulationStepSize;
                break;
            case 2:
                this.transform.position -= transform.forward * move_Speed* simulation.SimulationStepSize;
                break;
            case 3:
                transform.Rotate(new Vector3(0, -1, 0) * rotate_Speed* simulation.SimulationStepSize, Space.World);
                break;
            case 4:
                transform.Rotate(new Vector3(0, 1, 0) * rotate_Speed* simulation.SimulationStepSize, Space.World);
                break;
        }

        simulation.Simulate();

        step += 1;

        if(alga_remaining == 0 || step >= max_step){
            done = true;
        }
        
        x = this.transform.position.x;
        z = this.transform.position.z;
        if(x<-boat_size_x || x>boat_size_x || z<-boat_size_z || z>boat_size_z){
            done = true;
            reward = -100;
        }

        return new RlResult(reward, done, GetObservation());
    }

    public RlReset Reset(){
        boat_size = 10;
        float boat_dilate_x = Random.Range(0.7f, 1.3f);
        float boat_dilate_z = Random.Range(0.7f, 1.3f);
        boat_size_x = boat_size*boat_dilate_x;
        boat_size_z = boat_size*boat_dilate_z;
        boat_size_x -= boat_size_x % alga_resolution;
        boat_size_z -= boat_size_z % alga_resolution;
        boat.transform.localScale  = new Vector3(2*boat_size_x, 1, 2*boat_size_z);

        float res_inverse = 1/(float)alga_resolution;
        alga_remaining = 0;

        GameObject[] algas = GameObject.FindGameObjectsWithTag("Alga");
        foreach(GameObject alga in algas){
            GameObject.Destroy(alga);
        }

        for (int i = (int) -boat_size_x* alga_resolution; i < (int) boat_size_x*alga_resolution ; i++)
        {
            for (int j = (int) -boat_size_z* alga_resolution; j < (int) boat_size_z * alga_resolution; j++)
            {
                spawn_position = new Vector3(i* res_inverse + res_inverse / 2, 0.03f, j* res_inverse + res_inverse / 2);
                spawner = Instantiate(alga_prefab, spawn_position, Quaternion.identity);
                spawner.transform.localScale = new Vector3(res_inverse, 0.06f, res_inverse);
                spawner.tag = "Alga";
                alga_remaining += 1;
            }
        }

        alga_init_count = alga_remaining;

        float init_x = Random.Range(-boat_size_x, boat_size_x);
        float init_z = Random.Range(-boat_size_z, boat_size_z);
        this.transform.position = new Vector3(init_x, 0.3f, init_z);

        float init_rotation = Random.Range(0.0f,360.0f);
        transform.Rotate(new Vector3(0, init_rotation, 0));
        
        done = false;
        step = 0;

        return GetResetObservation();
    }


    public RlReset GetResetObservation(){
        
        cam.Render();
        Texture2D snapshot = new Texture2D(resWidth, resHeight, TextureFormat.RGB24,false);            
        RenderTexture.active = cam.targetTexture;
        snapshot.ReadPixels(new Rect(0,0,resWidth,resHeight), 0, 0);
        Color32[] pixels = snapshot.GetPixels32(0);

        return new RlReset(alga_init_count, pixels, resWidth, resHeight);
    }

    public ImageObs GetObservation(){
        
        cam.Render();
        Texture2D snapshot = new Texture2D(resWidth, resHeight, TextureFormat.RGB24,false);            
        RenderTexture.active = cam.targetTexture;
        snapshot.ReadPixels(new Rect(0,0,resWidth,resHeight), 0, 0);
        Color32[] pixels = snapshot.GetPixels32(0);

        return new ImageObs(pixels, resWidth, resHeight);
    }


    void OnTriggerEnter(Collider other)
    {
        if(other.gameObject.layer == 6)
        {
            Destroy(other.gameObject);
            reward += 1;
            alga_remaining -=1;
        }
    }
}
