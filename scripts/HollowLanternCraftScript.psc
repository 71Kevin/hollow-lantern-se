Scriptname HollowLanternCraftScript extends ObjectReference

Light Property HollowLantern Auto
MiscObject Property HollowLanternToken Auto

Event OnContainerChanged(ObjectReference akNewContainer, ObjectReference akOldContainer)
	If akNewContainer
		akNewContainer.AddItem(HollowLantern, 1, true)
		akNewContainer.RemoveItem(HollowLanternToken, 1, true)
	EndIf
EndEvent
