using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CameraObservation : MonoBehaviour
{

    Camera cam;
    int resWidth = 128;
    int resHeight = 128;
    bool take = false;

    // Start is called before the first frame update
    void Awake()
    {
        cam = GetComponent<Camera>();
        if(cam.targetTexture == null){
            cam.targetTexture = new RenderTexture(resWidth, resHeight,0);
        }
        // cam.gameObject.SetActive(false);
    }

    // Update is called once per frame
    void Update()
    {
        if(Input.GetKeyDown(KeyCode.Space)){
            take = true;
        }
    }


    void LateUpdate(){
        // if(cam.gameObject.activeSelf){
        if(take){
            take = false;

            Texture2D snapshot = new Texture2D(resWidth, resHeight, TextureFormat.RGB24,false);            
            RenderTexture.active = cam.targetTexture;
            snapshot.ReadPixels(new Rect(0,0,resWidth,resHeight), 0, 0);
            Color[] pixels = snapshot.GetPixels(0);

            Debug.Log(pixels.GetLength(0));
            Debug.Log(pixels[0][0]);
            Debug.Log(pixels[0][1]);
            Debug.Log(pixels[0][2]);
        }
    }
}
