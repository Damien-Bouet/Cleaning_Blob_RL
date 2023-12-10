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
    public float move_Speed = 20f;
    public float rotate_Speed = 200f;

    private float y;
    // private float z;

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

    public static int alga_remaining;
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
        void Gravity(){  //nothing:0, forward:1, backward:2, left:3, right:4
            agent.Gravity();
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
    public int max_step = 2000;

    public float gravity_force = 0.01f;
    private Vector3 gravity_direction;
    private Rigidbody _rigidbody;


    // Start is called before the first frame update
    void Start()
    {
        simulation = GetComponent<Simulation>();
        _rigidbody = GetComponent<Rigidbody>();
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
                transform.Rotate(new Vector3(0, -1, 0) * rotate_Speed* simulation.SimulationStepSize, Space.Self);
                reward += 0.5f;
                break;
            case 4:
                transform.Rotate(new Vector3(0, 1, 0) * rotate_Speed* simulation.SimulationStepSize, Space.Self);
                reward += 0.5f;
                break;

        }


        for(var i=0; i<10; i++){
            // if(Mathf.Abs(transform.position.x) > boat_size*2){
            // gravity_direction = new Vector3(transform.position.x - boat_size*2*Mathf.Sign(transform.position.x),transform.position.y,transform.position.z);
            // }
            // else{
            gravity_direction = new Vector3(0,transform.position.y,transform.position.z);
            // }
            gravity_direction.Normalize();
            _rigidbody.AddForce(-gravity_direction*gravity_force);
            
            simulation.Simulate();
        }

        step += 1;

        if(action == 1){
            reward *= 1.5f;
        }
        
        if(alga_remaining == 0 || step >= max_step){
            done = true;
        }
        
        y = this.transform.position.y;
        // z = this.transform.position.z;
        // if(x<-boat_size_x || x>boat_size_x || z<-boat_size_z || z>boat_size_z){
        if(y<=0){
            done = true;
            reward = -100;
        }

        return new RlResult(reward , done, GetObservation());
    }

    public RlReset Reset(){
        float d = 0.5f;
        boat_size = 9.98f * d;
        boat_size_x = (boat_size+0.02f*d) * 2;
        // float boat_dilate_x = Random.Range(0.7f, 1.3f);
        // float boat_dilate_z = Random.Range(0.7f, 1.3f);
        // boat_size_x = boat_size*boat_dilate_x;
        // boat_size_z = boat_size*boat_dilate_z;
        // boat_size_x -= boat_size_x % alga_resolution;
        // boat_size_z -= boat_size_z % alga_resolution;
        // boat.transform.localScale  = new Vector3(2*boat_size_x, 1, 2*boat_size_z);

        float res_inverse = 1/(float)alga_resolution;

        alga_remaining = 0;

        GameObject[] algas = GameObject.FindGameObjectsWithTag("Alga");
        foreach(GameObject alga in algas){
            GameObject.Destroy(alga);
        }

    
        float d_theta = res_inverse/boat_size/1.05f;
        for (int i = (int) -Mathf.Ceil(boat_size_x* alga_resolution); i < (int) Mathf.Ceil(boat_size_x* alga_resolution) ; i++)
        {
            for (int j = (int) - Mathf.Ceil(boat_size*alga_resolution*Mathf.PI/2 * 1.05f); j < (int) Mathf.Ceil(boat_size*alga_resolution*Mathf.PI/2 * 1.05f); j++)
            {
                spawn_position = new Vector3(i* res_inverse + res_inverse / 2, boat_size * Mathf.Cos(d_theta*(j+0.5f)), boat_size * Mathf.Sin(d_theta*(j+0.5f)));
                spawner = Instantiate(alga_prefab, spawn_position, Quaternion.identity);
                spawner.transform.localScale = new Vector3(res_inverse, 0.06f, res_inverse);
                spawner.transform.eulerAngles = new Vector3(d_theta*(j+0.5f)*180/Mathf.PI, 0, 0);
                spawner.tag = "Alga";
                alga_remaining += 1;
            }
        }
        float d_phi = res_inverse/boat_size/1.05f;
        float r;
        for (int i = (int) 0; i < (int) Mathf.Ceil(boat_size*alga_resolution*Mathf.PI/2 * 1.05f) ; i++)
        {
            r = i/(Mathf.Ceil(boat_size*alga_resolution*Mathf.PI/2-1))*0.15f + 1.05f;
            d_theta = res_inverse/(boat_size* Mathf.Cos(d_phi*(i+0.5f)))/r;
            for (int j = (int) -Mathf.Ceil(boat_size* Mathf.Cos(d_phi*(i+0.5f)) *alga_resolution*Mathf.PI/2 * r); j < (int) Mathf.Ceil(boat_size* Mathf.Cos(d_phi*(i+0.5f)) *alga_resolution*Mathf.PI/2 * r); j++)
            {
                spawn_position = new Vector3(boat_size_x + boat_size*Mathf.Sin(d_phi*(i+0.5f)), boat_size* Mathf.Cos(d_phi*(i+0.5f)) * Mathf.Cos(d_theta*(j+0.5f)), boat_size * Mathf.Cos(d_phi*(i+0.5f)) * Mathf.Sin(d_theta*(j+0.5f)));
                spawner = Instantiate(alga_prefab, spawn_position, Quaternion.identity);
                spawner.transform.localScale = new Vector3(res_inverse, 0.06f, res_inverse);
                spawner.transform.eulerAngles = new Vector3(d_theta*(j+0.5f)*180/Mathf.PI, 0, -d_phi*(i+0.5f)*180/Mathf.PI);
                spawner.tag = "Alga";
                alga_remaining += 1;

                spawn_position = new Vector3(-boat_size_x - boat_size*Mathf.Sin(d_phi*(i+0.5f)), boat_size* Mathf.Cos(d_phi*(i+0.5f)) * Mathf.Cos(d_theta*(j+0.5f)), boat_size * Mathf.Cos(d_phi*(i+0.5f)) * Mathf.Sin(d_theta*(j+0.5f)));
                spawner = Instantiate(alga_prefab, spawn_position, Quaternion.identity);
                spawner.transform.localScale = new Vector3(res_inverse, 0.06f, res_inverse);
                spawner.transform.eulerAngles = new Vector3(d_theta*(j+0.5f)*180/Mathf.PI, 0, d_phi*(i+0.5f)*180/Mathf.PI);
                spawner.tag = "Alga";
                alga_remaining += 1;
            }
        }

        alga_init_count = alga_remaining;

        float init_x = Random.Range(-boat_size_x, boat_size_x);
        float init_theta = Random.Range(-Mathf.PI/2, Mathf.PI/2);
        float init_phi = Random.Range(0f,360f);
        this.transform.position = new Vector3(init_x, (boat_size+0.3f)*Mathf.Cos(init_theta), (boat_size+0.3f)*Mathf.Sin(init_theta));

        // float init_rotation = Random.Range(0.0f,360.0f);
        this.transform.eulerAngles = new Vector3(init_theta*180/Mathf.PI, 0, 0);
        this.transform.Rotate(new Vector3(0, 1, 0) * init_phi, Space.Self);
        done = false;
        step = 0;
        
        // for(int i = 0; i < 100; i++){
        //     algas = GameObject.FindGameObjectsWithTag("Alga");
        //     foreach(GameObject alga in algas){
                
        //         if(Mathf.Abs(alga.transform.position.x) > 20){
        //             gravity_direction = new Vector3(alga.transform.position.x - 20*Mathf.Sign(alga.transform.position.x),alga.transform.position.y,alga.transform.position.z);
        //         }
        //         else{
        //             gravity_direction = new Vector3(0,alga.transform.position.y,alga.transform.position.z);
        //         }
        //         gravity_direction.Normalize();
        //         alga.GetComponent<Rigidbody>().AddForce(-gravity_direction*gravity_force/100);
        //     }

        //     simulation.Simulate();
        // }


        simulation.Simulate();
        
        for(var i=0; i<5; i++){
            if(Mathf.Abs(transform.position.x) > boat_size*2){
            gravity_direction = new Vector3(transform.position.x - boat_size*2*Mathf.Sign(transform.position.x),transform.position.y,transform.position.z);
            }
            else{
                gravity_direction = new Vector3(0,transform.position.y,transform.position.z);
            }
            gravity_direction.Normalize();
            _rigidbody.AddForce(-gravity_direction*gravity_force);
            

            simulation.Simulate();
        }

        return GetResetObservation();
    }

    private void Gravity(){
        
        for(int i = 0; i < 10; i++){
            
            simulation.Simulate();
        }
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
            reward += 1 + (boat_size-other.transform.position.y)/boat_size;
            alga_remaining -=1;
        }
        
    }
}
