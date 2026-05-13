from typing import Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field


class ChurnInput(BaseModel):
    model_config = ConfigDict(populate_by_name=True, str_strip_whitespace=True)

    age: int = Field(..., ge=18, le=100, validation_alias=AliasChoices("Age", "age"))
    gender: Literal["Male", "Female"] = Field(
        ...,
        validation_alias=AliasChoices("Gender", "gender"),
    )
    tenure: int = Field(..., ge=0, le=72, validation_alias=AliasChoices("Tenure", "tenure"))
    usage_frequency: int = Field(
        ...,
        ge=0,
        validation_alias=AliasChoices("Usage Frequency", "Usage_Frequency", "usage_frequency"),
    )
    support_calls: int = Field(
        ...,
        ge=0,
        validation_alias=AliasChoices("Support Calls", "Support_Calls", "support_calls"),
    )
    payment_delay: int = Field(
        ...,
        ge=0,
        validation_alias=AliasChoices("Payment Delay", "Payment_Delay", "payment_delay"),
    )
    subscription_type: Literal["Basic", "Standard", "Premium"] = Field(
        ...,
        validation_alias=AliasChoices(
            "Subscription Type",
            "Subscription_Type",
            "subscription_type",
        ),
    )
    contract_length: Literal["Monthly", "Quarterly", "Annual"] = Field(
        ...,
        validation_alias=AliasChoices(
            "Contract Length",
            "Contract_Length",
            "contract_length",
        ),
    )
    total_spend: float = Field(
        ...,
        ge=0,
        validation_alias=AliasChoices("Total Spend", "Total_Spend", "total_spend"),
    )
    last_interaction: int = Field(
        ...,
        ge=0,
        validation_alias=AliasChoices(
            "Last Interaction",
            "Last_Interaction",
            "last_interaction",
        ),
    )

    def as_raw_features(self) -> dict[str, int | float | str]:
        return {
            "Age": self.age,
            "Gender": self.gender,
            "Tenure": self.tenure,
            "Usage Frequency": self.usage_frequency,
            "Support Calls": self.support_calls,
            "Payment Delay": self.payment_delay,
            "Subscription Type": self.subscription_type,
            "Contract Length": self.contract_length,
            "Total Spend": self.total_spend,
            "Last Interaction": self.last_interaction,
        }


class PredictionResponse(BaseModel):
    prediction: int
    status: Literal["Churn", "No Churn"]


class BatchPredictionResponse(BaseModel):
    total: int
    predictions: list[PredictionResponse]


class HealthResponse(BaseModel):
    status: Literal["ok", "error"]
    model_ready: bool
