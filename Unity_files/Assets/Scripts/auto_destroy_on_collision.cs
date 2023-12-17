using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class auto_destroy_on_collision : MonoBehaviour
{
    public static int alga_remaining;
    
    void OnTriggerStay(Collider other)
    {
        if(other.gameObject.layer == 6)
        {
            Destroy(other.gameObject);
            alga_remaining -=1;
            Debug.Log(alga_remaining);
        }
        
    }
}
